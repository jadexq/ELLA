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
        self.rho = 0.0
        # precompute constant per-bin geometry (#3): r_mid and the beta kernel varphi
        # depend only on the fixed n_bins / a0 / b0, so compute once here instead of
        # rebuilding (and calling scipy) every forward. Registered as buffers so they
        # follow the module's device/dtype.
        r_mid = (torch.arange(self.n_bins, dtype=torch.float32) + 0.5) / self.n_bins
        varphi = torch.as_tensor(
            beta_dist.pdf(r_mid.numpy(), self.a0, self.b0), dtype=torch.float32
        )  #!!! beta kernel
        # persistent=False: keep them out of state_dict (they are constants
        # recomputed in __init__; on_train_end assumes state_dict holds only the
        # scalar sqrt_* params) while still following the module's device.
        self.register_buffer("r_mid", r_mid, persistent=False)
        self.register_buffer("varphi", varphi, persistent=False)

    def forward(self, sc_totals) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        mu:    [num_cell, num_bin]  per-(cell, bin) Normal mean
        sigma: [num_bin]            per-bin Normal std (broadcasts over cells)
        """
        # vectorized over (cells x bins): one tensor op replaces the per-(cell, bin)
        # Python loop (#2). Same math as the loop, using r_mid / varphi precomputed
        # in __init__ (#3).
        alpha = torch.square(self.sqrt_alpha)
        beta = torch.square(self.sqrt_beta)
        sigma02 = torch.square(self.sqrt_sigma02)
        sc_total = sc_totals / 1.0  # [n_cells]
        # mu[c, b] = (alpha + beta * varphi[b]) * sc_total[c] * 2*pi * r_mid[b]
        mu = (
            (alpha + beta * self.varphi)[None, :]
            * sc_total[:, None]
            * 2 * math.pi
            * self.r_mid[None, :]
        )  # [n_cells, n_bins]
        sigma = torch.sqrt(sigma02 + self.rho * self.varphi)  # [n_bins]
        return mu, sigma


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

    def get_lambda_star_i(self, cells_sc_total) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        lam:      [num_cell, num_bin]  sampled lambda* (post-relu)
        log_prob: [num_cell, num_bin]  log_prob of the pre-relu sample
        """
        # vectorized over (cells x bins): one batched Normal + a single rsample()
        # replaces the per-(cell, bin) distribution construction loop (#2).
        mu, sigma = self.policy_net(cells_sc_total)  # [n_cells, n_bins], [n_bins]
        normal_dist = torch.distributions.Normal(mu, sigma)
        lambda_star_i = normal_dist.rsample()  # [n_cells, n_bins]
        log_prob = normal_dist.log_prob(lambda_star_i)  # on the pre-relu sample
        lam = F.relu(lambda_star_i)
        return lam, log_prob

    def forward(self, batch) -> Tuple[Tensor, Tensor]:  # pylint: disable=arguments-differ
        cells_points, cells_points_length, cells_sc_total = batch
        lam, log_prob = self.get_lambda_star_i(cells_sc_total)  # [n_cells, n_bins] each
        cells_log_weight = []  # exp argument per cell: cell_reward - log(n!)
        cells_sum_log_prob = []
        cells_reward = []
        for cell_i, (cell_points, cell_points_length) in enumerate(
            zip(cells_points, cells_points_length)
        ):
            lam_vec = lam[cell_i]  # [n_bins]: sampled lambda* per radial bin
            int_lambda_star_i = torch.mean(lam_vec)
            sum_log_prob = torch.sum(log_prob[cell_i])
            cell_points = cell_points[:cell_points_length]
            if len(cell_points) > 0: # if this cell has >0 num of points
                # vectorized per-transcript bin lookup (replaces the per-molecule loop):
                # bin each point's normalized radius, clamp points>1.0 to the last bin,
                # gather lam* and sum log. Same math as the loop, one gather instead of N.
                bin_idx = torch.clamp(
                    torch.floor(cell_points * self.n_bins).long(), max=self.n_bins - 1
                )  #!!! clamp points>1.0 to 1.0
                sum_log_lambda_star_i_point = torch.log(lam_vec[bin_idx] + 1e-10).sum()
                cell_reward = -int_lambda_star_i + sum_log_lambda_star_i_point
                cells_log_weight.append(cell_reward - math.lgamma(len(cell_points) + 1))
                cells_sum_log_prob.append(sum_log_prob)
                cells_reward.append(cell_reward)
        # Numerical stabilization (log-sum-exp trick): the per-cell weight is
        # exp(cell_reward - log n!), whose argument grows with transcript count and
        # overflows to inf in float32 (-> nan params -> Normal() crash). Subtract a
        # DETACHED per-batch max before exp so the largest term is exp(0)=1. Because
        # `m` carries no gradient, this only rescales the loss by the constant exp(-m):
        # the gradient direction, the likelihood, and `reward` (used downstream for the
        # LRT statistic / kernel weights) are all unchanged.
        log_weight = torch.stack(cells_log_weight)
        m = log_weight.max().detach()
        cells_loss = torch.stack(cells_sum_log_prob) * torch.exp(log_weight - m)
        loss = -torch.mean(cells_loss)
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
