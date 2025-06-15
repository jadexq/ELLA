import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from scipy.stats import beta as beta_dist
from torch import Tensor, nn, optim
from torch.optim import Optimizer

from ella.options import KernelParam, ModelConfig


class PolicyNetwork(nn.Module):  # pylint: disable=too-many-instance-attributes
    def __init__(self, cfg: ModelConfig, init_values: Dict, kernel_param: KernelParam) -> None:
        super().__init__()
        self.n_bins = cfg.n_bins
        self.a0 = kernel_param.a0
        self.b0 = kernel_param.b0
        sqrt_alpha_init = math.sqrt(init_values["lam_null"]) 
        # sqrt_alpha_init = 1.21 # <<<!!! fix alpha init
        sqrt_beta_init = math.sqrt(cfg.beta_init)
        sqrt_sigma02_init = math.sqrt(cfg.sigma02_init)
        self.sqrt_alpha = nn.Parameter(torch.tensor(sqrt_alpha_init))  #!!!
        self.sqrt_beta = nn.Parameter(torch.tensor(sqrt_beta_init), requires_grad=cfg.beta_requires_grad)  #!!!
        self.sqrt_sigma02 = nn.Parameter(torch.tensor(sqrt_sigma02_init))
        # self.log_rho = nn.Parameter(torch.randn(1) * 0.01)
  

    def forward(self, sc_totals) -> List[List[Tuple[Tensor, Tensor]]]:
        """
        Returns
        -------
        [[(mu, sigma), ...], ...]
        shape: num_cell * num_bin
        """
        cells_stats: List[List[Tuple[Tensor, Tensor]]] = []
        alpha = torch.square(self.sqrt_alpha)
        beta = torch.square(self.sqrt_beta)
        sigma02 = torch.square(self.sqrt_sigma02)
        # sigma02 = torch.tensor(0.0001)
        # rho = torch.exp(self.log_rho)
        rho = 0.0
        for sc_total in sc_totals:
            bins_stats: List[Tuple[Tensor, Tensor]] = []
            # sc_total = sc_total/1000
            sc_total = sc_total/1.0
            for bin_idx in range(self.n_bins):
                r_mid = (bin_idx + 0.5) / self.n_bins # using r_mid
                if True:
                    varphi = beta_dist.pdf(r_mid, self.a0, self.b0) #!!! beta kernel
                if False:
                    # varphi = indicator(r_mid, self.a0, self.b0) #!!! stepwise kernel
                    varphi = 1 if self.a0 <= r_mid <= self.b0 else 0
                mu = (alpha + beta * varphi) * sc_total * 2 * math.pi * r_mid
                # r_right = (bin_idx + 1) / self.n_bins # using r_right
                # varphi = beta_dist.pdf(r_right, self.a0, self.b0)
                # mu = (alpha + beta * varphi) * sc_total * 2 * math.pi * r_right
                # sigma = torch.sqrt(sigma02 + rho * varphi) * sc_total * 2 * math.pi * r_mid # not using this
                sigma = torch.sqrt(sigma02 + rho * varphi)
                bin_stats: Tuple[Tensor, Tensor] = (mu, sigma)
                bins_stats.append(bin_stats)
            cells_stats.append(bins_stats)
        return cells_stats


class COX(L.LightningModule):
    def __init__(
        self,
        cfg: ModelConfig,
        init_values: Dict,
        kernel_idx: int,
        kernel_param: KernelParam,
    ) -> None:
        super().__init__()
        self.cfg: ModelConfig = cfg
        self.kernel_idx = kernel_idx
        self.kernel_param = kernel_param
        self.n_bins: int = cfg.n_bins
        self.policy_net = PolicyNetwork(cfg=cfg, init_values=init_values, kernel_param=kernel_param)
        self.epoch_infos: List[Dict] = []

    def get_lambda_star_i(self, cells_sc_total) -> List[List[Tuple[Tensor, Tensor]]]:
        """
        Returns
        -------
        [[(sample, log_prob), ...], ...]
        shape: num_cell * num_bin
        """
        all_lambda_star_i: List[List[Tuple[Tensor, Tensor]]] = []
        cells_stats: List[List[Tuple[Tensor, Tensor]]] = self.policy_net(cells_sc_total)
        for bins_stats in cells_stats:
            cell_lambda_star_i: List[Tuple[Tensor, Tensor]] = []
            for bin_stats in bins_stats:
                mu, sigma = bin_stats
                normal_dist = torch.distributions.Normal(mu, sigma)
                lambda_star_i = normal_dist.rsample()
                log_prob_lambda_star_i = normal_dist.log_prob(lambda_star_i)
                lambda_star_i = F.relu(lambda_star_i)
                cell_lambda_star_i.append((lambda_star_i, log_prob_lambda_star_i))
            all_lambda_star_i.append(cell_lambda_star_i)
        return all_lambda_star_i

    def forward(self, batch) -> Tuple[Tensor, Tensor]:  # pylint: disable=arguments-differ
        cells_points, cells_points_length, cells_sc_total = batch
        cells_lambda_star_i: List[List[Tuple[Tensor, Tensor]]] = self.get_lambda_star_i(cells_sc_total)
        cells_loss = []
        cells_reward = []
        for cell_lambda_star_i, cell_points, cell_points_length in zip(
            cells_lambda_star_i, cells_points, cells_points_length
        ):
            int_lambda_star_i = torch.mean(torch.stack([x[0] for x in cell_lambda_star_i]).squeeze(-1))
            sum_log_prob = torch.sum(torch.stack([x[1] for x in cell_lambda_star_i]))
            lambda_star_i_points_log = []
            cell_points = cell_points[:cell_points_length]
            if len(cell_points) > 0: # if this cell has >0 num of points
                for point in cell_points:
                    bin_idx = torch.floor(point * self.n_bins).to(torch.int)
                    bin_idx = torch.minimum(bin_idx, torch.tensor(self.n_bins-1)) #!!! equavalent to clamp points>1.0 to 1.0
                    lambda_star_i_point, _lambda_star_i_point_log_prob = cell_lambda_star_i[bin_idx]
                    lambda_star_i_points_log.append(torch.log(lambda_star_i_point + 1e-10))
                sum_log_lambda_star_i_point = torch.sum(torch.stack(lambda_star_i_points_log))
                cell_reward = -int_lambda_star_i + sum_log_lambda_star_i_point
                # cell_loss = sum_log_prob * cell_reward
                const = 0.0
                if False: # currently don't use this
                    if len(cell_points) > 200:
                        const = 80.0
                cell_loss = sum_log_prob * torch.exp(cell_reward - math.lgamma(len(cell_points) + 1) - const)
                cells_reward.append(cell_reward)
                cells_loss.append(cell_loss)
        loss = -torch.mean(torch.stack(cells_loss))
        reward = torch.sum(torch.stack(cells_reward))
        return loss, reward

    def training_step(self, batch, _batch_idx) -> Tensor:  # pylint: disable=arguments-differ
        loss, reward = self(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_reward", reward, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        return optim.Adam(self.policy_net.parameters(), lr=self.cfg.optimizer.lr)

    def on_train_epoch_end(self) -> None:
        if self.cfg.is_debug:
            epoch: int = self.trainer.current_epoch
            loss_tensor = self.trainer.callback_metrics.get("train_loss")
            assert loss_tensor
            reward_tensor = self.trainer.callback_metrics.get("train_reward")
            assert reward_tensor
            self.epoch_infos.append(
                {
                    "epoch": epoch,
                    "loss": loss_tensor.item(),
                    "reward": reward_tensor.item(),
                }
            )

    def on_train_end(self) -> None:
        assert self.logger and self.logger.log_dir

        best_checkpoint = torch.load(self.trainer.checkpoint_callback.best_model_path)
        best_model_state_dict = best_checkpoint['state_dict']

        params_path = Path(self.logger.log_dir, "model_params.json")
        params = {k: v.tolist() for k, v in best_model_state_dict.items()}
        final_params = {f"{k.split('_')[-1]}": v * v for k, v in params.items()}
        params |= final_params
        with params_path.open("w", encoding="utf-8") as f:
            json.dump(params, f, indent=4, sort_keys=False)

        best_epoch_info = max(self.epoch_infos, key=lambda x: x["reward"])
        result_path = Path(self.logger.log_dir, "result.json")
        result: Dict = {
            "alpha": params["alpha"],
            "beta": params["beta"],
            "reward": best_epoch_info["reward"],
            "a0": self.kernel_param.a0,
            "b0": self.kernel_param.b0,
            "kernel_idx": self.kernel_idx,
        }
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        if self.cfg.is_debug:
            loss_path = Path(self.logger.log_dir, "epoch_infos.jsonl")
            with loss_path.open("w", encoding="utf-8") as f:
                for epoch_info in self.epoch_infos:
                    f.write(f"{json.dumps(epoch_info)}\n")
