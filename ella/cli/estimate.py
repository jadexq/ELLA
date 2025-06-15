import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.stats import beta as beta_dist

from ella.options import EstimationConfig


@dataclass
class TrainingResult:
    alpha: float
    beta: float
    reward: float
    a0: float
    b0: float
    kernel_idx: int
        

def indicator(x, a, b):
    return 1 if a <= x <= b else 0        


def estimate(cfg: EstimationConfig, log_dirs: List[Path]) -> None:
    training_results: List[TrainingResult] = []
    reward_null: Optional[float] = None
    for log_dir in log_dirs:
        with (log_dir / "result.json").open("r", encoding="utf-8") as f:
            training_result = TrainingResult(**json.load(f))
            if training_result.a0 == 1.0 and training_result.b0 == 1.0:
                assert reward_null is None, "Found kernel null twice"
                reward_null = training_result.reward
            else:
                training_results.append(training_result)
    with (log_dirs[0] / "dataset_info.json").open("r", encoding="utf-8") as f:
        dataset_info: Dict = json.load(f)
        gene_id: str = dataset_info["gene_id"]
    assert reward_null is not None, "Not found kernel null"
    training_results.sort(key=lambda x: x.kernel_idx)

    lam_raw: List[List[float]] = []
    p_raw: List[float] = []
    for tr in training_results:
        lam_l: List[float] = []
        for bin_idx in range(cfg.n_bins):
            r_mid: float = (bin_idx + 0.5) / cfg.n_bins
            if True:
                varphi: float = beta_dist.pdf(r_mid, tr.a0, tr.b0) #!!! beta kernel
            if False:
                varphi: float = 1 if tr.a0 <= r_mid <= tr.b0 else 0 #!!! stepwise kernel
            mu: float = tr.alpha + tr.beta * varphi
            lam_l.append(mu)
        lam_raw.append(lam_l)

        test_stat = -2 * (reward_null - tr.reward)
        if not test_stat > 0.0:
            p = 1 - np.random.uniform(low=0.0, high=0.5)
        else:
            p = 1 - 0.5 - 0.5 * stats.chi2.cdf(test_stat, 1)
        p_raw.append(p)

    # cauchy combination
    num_of_kernels: int = len(training_results)
    w = np.full(num_of_kernels, 1 / num_of_kernels)  # equal weights
    p_raw_array = np.array(p_raw)
    p_raw_array[p_raw_array > 0.9999999] = 0.9999999
    tt = np.sum(w * np.tan(math.pi * (0.5 - p_raw_array)))
    p_cauchy = 0.5 - np.arctan(tt) / math.pi

    rewards = torch.tensor([x.reward for x in training_results])
    weights = F.softmax(rewards, dim=0).unsqueeze(-1)
    lam_raw_tensor = torch.tensor(lam_raw)
    lam_weighted = torch.sum(weights * lam_raw_tensor, dim=0)

    estimation_result: Dict = {
        "gene_id": gene_id,
        "p_raw": p_raw,
        "reward_null": reward_null,
        "rewards": rewards.tolist(),
        "weights": weights.squeeze(dim=-1).tolist(),
        "p_cauchy": p_cauchy,
        "lam_weighted": lam_weighted.tolist(),
    }

    estimation_result_path = Path(cfg.output_path)
    estimation_result_path.parent.mkdir(parents=True, exist_ok=True)
    with estimation_result_path.open("w", encoding="utf-8") as f:
        json.dump(estimation_result, f, indent=4)


def parse_args() -> EstimationConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--search-dir", type=str, required=True)
    parser.add_argument("-p", "--log-dir-pattern", type=str, required=True)
    parser.add_argument("-o", "--output-path", type=str, required=True)
    parser.add_argument("-b", "--n-bins", type=int, required=True)
    args = parser.parse_args()
    return EstimationConfig(**vars(args))


def main() -> None:
    cfg: EstimationConfig = parse_args()
    log_dir_pattern = re.compile(cfg.log_dir_pattern)
    log_dirs = [path for path in Path(cfg.search_dir).iterdir() if log_dir_pattern.search(str(path))]
    estimate(cfg=cfg, log_dirs=log_dirs)
