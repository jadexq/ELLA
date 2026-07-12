"""Benchmark step 1 -- null2 (100 genes): runtime, compute, QQ overlay.

Runs ELLA Newton and ELLA Adam on the null2 scaffold (uniform-in-registration-frame
null, so a calibrated method gives uniform p-values), 100 genes. Reports wall-clock
+ peak memory per arm, and a QQ plot of the raw per-gene Cauchy p-values vs Uniform,
both arms overlaid.

Run in the `ella1` conda env, on a compute node:
    conda run -n ella1 --no-capture-output python -u benchmark/step1_null2.py
"""
import os
# pin BLAS to 1 thread/worker BEFORE numpy import (inherited by forked workers)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner  # noqa: E402

DATA = Path("/scratch/user/jadewang/ella2d_v4/simulation/xenium/output/step3/data/null2_ng1000_mu5")
WORK = Path("/scratch/user/jadewang/ella2d_v4/tmp_ella1_eval/benchmark/null2")
OUT = Path(__file__).resolve().parent / "output"
N_JOBS = int(os.environ.get("N_JOBS", "64"))
MAX_GENES = 100
SEED_BASE = 1000
SOURCES = ("newton", "adam")


def make_qq(raw: pd.DataFrame, path: Path):
    """Uniform QQ on -log10 axes, both arms overlaid (expected vs observed p)."""
    colors = {"newton": "#2e7d32", "adam": "#c0362c"}
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=200)
    hi = 0.0
    for src in SOURCES:
        p = np.sort(raw.loc[raw.method == src, "pvalue"].to_numpy(float))
        p = p[np.isfinite(p)]
        n = len(p)
        expected = (np.arange(1, n + 1) - 0.5) / n
        x = -np.log10(expected)
        y = -np.log10(np.clip(p, 1e-300, 1.0))
        hi = max(hi, x.max(), y.max())
        ax.scatter(x, y, s=14, alpha=0.7, edgecolor="none",
                   color=colors[src], label=f"ELLA {src} (n={n})")
    lim = hi * 1.05
    ax.plot([0, lim], [0, lim], color="0.5", lw=1, ls="--", zorder=0)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(r"expected $-\log_{10}(p)$")
    ax.set_ylabel(r"observed $-\log_{10}(p)$")
    ax.set_title("null2 (100 genes): p-value calibration")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, raw = [], []
    for src in SOURCES:
        print(f"=== null2 / {src} ===")
        df, meta = runner.run_dataset(DATA, WORK / src, source=src, n_jobs=N_JOBS,
                                      seed_base=SEED_BASE, max_genes=MAX_GENES)
        df["method"] = src
        raw.append(df[["gene", "pvalue", "method"]])
        rows.append(dict(experiment="null2", **meta))

    raw = pd.concat(raw, ignore_index=True)
    raw.to_csv(OUT / "null2_raw.csv", index=False)

    # resources.csv is shared across steps; replace this experiment's rows
    res_path = OUT / "resources.csv"
    resdf = pd.DataFrame(rows)
    if res_path.exists():
        old = pd.read_csv(res_path)
        old = old[old["experiment"] != "null2"]
        resdf = pd.concat([old, resdf], ignore_index=True)
    resdf.to_csv(res_path, index=False)
    print(resdf.to_string(index=False))

    make_qq(raw, OUT / "null2_qq.png")
    print("null2 done.")


if __name__ == "__main__":
    main()
