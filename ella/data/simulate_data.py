import json
import random
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import beta as beta_dist


@dataclass
class Bin:
    mid: float
    left: float
    right: float


def generate_r_bins(n_bins: int) -> List[Bin]:
    assert n_bins > 0
    bins = []
    width = 1 / n_bins
    for i in range(n_bins):
        left = i * width
        right = (i + 1) * width
        mid = (left + right) / 2
        bins.append(Bin(mid=mid, left=left, right=right))
    return bins


def generate_lambda_star_i(  # pylint: disable=too-many-positional-arguments
    r_bins: List[Bin],
    a0: float,
    b0: float,
    alpha: float,
    beta: float,
    sigma02: float,
    rho: float,
    sc_total: float,
) -> List[float]:
    results: List[float] = []
    for r_bin in r_bins:
        r_mid = r_bin.mid
        varphi: float = beta_dist.pdf(r_mid, a0, b0)
        if sigma02 > 0:
            epsilon_i: float = np.random.normal(0, np.sqrt(sigma02 + rho * varphi))
        else:
            epsilon_i: float = 0.0
        # lambda_i: float = alpha + beta * varphi + epsilon_i
        lambda_i: float = alpha + beta * varphi 
        # lambda_star_i: float = sc_total * 2 * pi * r_mid * lambda_i
        lambda_star_i: float = (sc_total/1.0) * 2 * math.pi * r_mid * lambda_i + epsilon_i # CPT
        lambda_star_i = max(lambda_star_i, 1e-10)
        results.append(lambda_star_i)
    return results





def generate_lambda_star_i_stepwise(  # pylint: disable=too-many-positional-arguments
    r_bins: List[Bin],
    a0: float,
    b0: float,
    alpha: float,
    beta: float,
    sigma02: float,
    rho: float,
    sc_total: float,
) -> List[float]:
    results: List[float] = []
    for r_bin in r_bins:
        r_mid = r_bin.mid
        varphi: float = 1 if a0 <= r_mid <= b0 else 0
        if sigma02 > 0:
            epsilon_i: float = np.random.normal(0, np.sqrt(sigma02 + rho * varphi))
        else:
            epsilon_i: float = 0.0
        lambda_i: float = alpha + beta * varphi 
        lambda_star_i: float = (sc_total/1.0) * 2 * math.pi * r_mid * lambda_i + epsilon_i # CPT
        lambda_star_i = max(lambda_star_i, 1e-10)
        results.append(lambda_star_i)
    return results





# fix count per cell
def generate_data(
    count_per_cell: int,
    lambda_star_i: List[float],
    r_bins: List[Bin],
) -> List[float]:
    assert len(lambda_star_i) == len(r_bins)
    sum_lambda_star_i = sum(lambda_star_i)
    probs: List[float] = [x / sum_lambda_star_i for x in lambda_star_i]
    counts = np.random.multinomial(count_per_cell, probs)
    points: List[float] = []
    for r_bin, count in zip(r_bins, counts):
        points.extend([random.uniform(r_bin.left, r_bin.right) for _ in range(count)])
    return points



def generate_data_stepwise(
    count_per_cell: int,
    lambda_star_i: List[float],
    r_bins: List[Bin],
) -> List[float]:
    assert len(lambda_star_i) == len(r_bins)
    sum_lambda_star_i = sum(lambda_star_i)
    probs: List[float] = [x / sum_lambda_star_i for x in lambda_star_i]
    counts = np.random.multinomial(count_per_cell, probs)
    points: List[float] = []
    for r_bin, count in zip(r_bins, counts):
        points.extend([random.uniform(r_bin.left, r_bin.right) for _ in range(count)])
    return points



# free count per cell
# def generate_data(
#     count_per_cell: int, # not in use
#     lambda_star_i: List[float],
#     r_bins: List[Bin],
# ) -> List[float]:
#     assert len(lambda_star_i) == len(r_bins)
#     counts = []
#     for r_bin, lam in zip(r_bins, lambda_star_i):
#         prob = (r_bin.right - r_bin.left) * lam
#         count = np.random.poisson(prob, size=None)
#         counts.append(count)
#     points: List[float] = []
#     for r_bin, count in zip(r_bins, counts):
#         points.extend([random.uniform(r_bin.left, r_bin.right) for _ in range(count)])
#     return points



def main(data_path: str = "simulated_data.jsonl") -> None:
    n_genes: int = 1 # <<<
    n_cells: int = 50 # <<<
    SC_TOTAL: float = 5.0 # fixed/tmp sc total (will update later) !!!!!
    genes_data: List[Dict] = []
    for idx_gene in range(n_genes):
        gene_data: Dict = {"gene_id": idx_gene}
        gene_data["cells"] = []
        sum_ci = 0
        sum_counts = 0
        for idx_cell in range(n_cells):
            cell_info: Dict = {"cell_id": idx_cell}
            r_bins: List[Bin] = generate_r_bins(15) # <<< #5
            if False:
                SC_TOTAL = np.random.negative_binomial(1, 0.16) # NB(n,p) # varying sc total !!!!!
            if True:
                lambda_star_i: List[float] = generate_lambda_star_i( #!!! beta
                    r_bins=r_bins,
                    a0=10.0, # <<<
                    b0=1.0, # <<<
                    alpha=0.0, # <<<
                    beta=1.0, # <<< 
                    sigma02=0.0, # <<<
                    rho=0.0,
                    sc_total=SC_TOTAL,
                )
            if False:
                lambda_star_i: List[float] = generate_lambda_star_i_stepwise( #!!! stepwise
                    r_bins=r_bins,
                    a0=0.6, # <<<
                    b0=0.8, # <<<
                    alpha=1.0, # <<<
                    beta=0.5, # <<< 
                    sigma02=0.0, # <<<
                    rho=0.0,
                    sc_total=SC_TOTAL,
                )
                    
            cell_info["points"] = generate_data(
                count_per_cell=SC_TOTAL, 
                lambda_star_i=lambda_star_i,
                r_bins=r_bins,
            )
            cell_info["sc_total"] = SC_TOTAL
            gene_data["cells"].append(cell_info)

            sum_ci = sum_ci + SC_TOTAL/1.0 #!!!
            sum_counts = sum_counts + len(cell_info["points"])
        genes_data.append(gene_data)
        gene_data["lam_null"] = sum_counts / (math.pi * sum_ci)
    # didn't sort genes by counts, bc in simu, counts are similar


    out_path = Path(data_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with out_path.open("w", encoding="utf-8") as f:
        for gene_data in genes_data:
            f.write(f"{json.dumps(gene_data)}\n")


if __name__ == "__main__":
    main()
