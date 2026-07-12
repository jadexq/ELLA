"""Benchmark step 2 -- cyto_symmetric (100 genes): runtime, compute, power, KLD.

Runs ELLA Newton and ELLA Adam on the cyto_symmetric scaffold (pure-radial planted
pattern peaking mid-cytoplasm, KL=0.02), 100 genes. Reports per arm:
  * wall-clock + peak memory,
  * power = fraction of genes with BH-FDR(pv_cauchy) < 0.05,
  * accuracy = per-gene KL(true || estimated radial profile) in nats, where `true`
    is the planted cyto_symmetric per-area radial intensity (output/cyto_truth.csv,
    precomputed on ELLA's linspace(0.001,0.999,100) grid) and `est` is ELLA's
    weighted_lam_est. Each length-100 curve is normalized to a pmf on the grid.

Run in the `ella1` conda env, on a compute node:
    conda run -n ella1 --no-capture-output python -u benchmark/step2_cyto_symmetric.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner  # noqa: E402

DATA = Path("/scratch/user/jadewang/ella2d_v4/simulation/xenium/output/step3/data/cyto_symmetric_skl0.02_ng100_mu5")
WORK = Path("/scratch/user/jadewang/ella2d_v4/tmp_ella1_eval/benchmark/cyto_symmetric")
OUT = Path(__file__).resolve().parent / "output"
N_JOBS = int(os.environ.get("N_JOBS", "64"))
SEED_BASE = 1000
SIG = 0.05
SOURCES = ("newton", "adam")
_EPS = 1e-12


def _pmf(v):
    v = np.clip(np.asarray(v, float), _EPS, None)
    return v / v.sum()


def kl_true_est(lam_true, lam_est):
    """KL(true || est) in nats, both normalized to a pmf on the grid."""
    p = _pmf(lam_true)
    q = _pmf(lam_est)
    return float(np.sum(p * np.log(p / q)))


def make_kld_plot(kld: pd.DataFrame, path: Path):
    colors = {"newton": "#2e7d32", "adam": "#c0362c"}
    fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=200)
    data = [kld.loc[kld.method == s, "kl"].to_numpy() for s in SOURCES]
    bp = ax.boxplot(data, labels=[f"ELLA {s}" for s in SOURCES],
                    widths=0.5, showfliers=False, patch_artist=True)
    for patch, s in zip(bp["boxes"], SOURCES):
        patch.set_facecolor(colors[s]); patch.set_alpha(0.35)
    for med in bp["medians"]:
        med.set_color("black")
    # jittered points
    rng = np.random.default_rng(0)
    for i, (s, d) in enumerate(zip(SOURCES, data), start=1):
        x = i + (rng.random(len(d)) - 0.5) * 0.25
        ax.scatter(x, d, s=10, alpha=0.5, edgecolor="none", color=colors[s])
    ax.set_ylabel(r"recovery KL(true $\parallel$ est)  [nats]")
    ax.set_title("cyto_symmetric (100 genes): profile accuracy")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(OUT / "cyto_truth.csv")
    lam_true = truth["lam_true"].to_numpy(float)

    rows, power_rows, kld_rows, profiles = [], [], [], []
    for src in SOURCES:
        print(f"=== cyto_symmetric / {src} ===")
        df, meta = runner.run_dataset(DATA, WORK / src, source=src, n_jobs=N_JOBS,
                                      seed_base=SEED_BASE, max_genes=None)
        rows.append(dict(experiment="cyto_symmetric", **meta))

        # power: BH-FDR on the pooled raw p-values
        p = df["pvalue"].to_numpy(float)
        fdr = false_discovery_control(p, method="bh")
        n_sig = int(np.sum(fdr < SIG))
        power = n_sig / len(p)
        power_rows.append(dict(method=src, n_genes=len(p), n_sig=n_sig, power=round(power, 4)))
        print(f"  power({src}) = {n_sig}/{len(p)} = {power:.3f}")

        # accuracy: per-gene KL(true || est)
        for _, r in df.iterrows():
            lam_est = np.asarray(r["lam_weighted"], float)
            kld_rows.append(dict(method=src, gene=r["gene"],
                                 kl=kl_true_est(lam_true, lam_est)))
            profiles.append(dict(method=src, gene=r["gene"],
                                 lam_weighted=r["lam_weighted"]))

    # write resources (shared file; replace this experiment's rows)
    res_path = OUT / "resources.csv"
    resdf = pd.DataFrame(rows)
    if res_path.exists():
        old = pd.read_csv(res_path)
        old = old[old["experiment"] != "cyto_symmetric"]
        resdf = pd.concat([old, resdf], ignore_index=True)
    resdf.to_csv(res_path, index=False)

    power_df = pd.DataFrame(power_rows)
    power_df.to_csv(OUT / "cyto_power.csv", index=False)
    kld_df = pd.DataFrame(kld_rows)
    kld_df.to_csv(OUT / "cyto_kld.csv", index=False)
    import json
    pd.DataFrame(profiles).assign(
        lam_weighted=lambda d: d["lam_weighted"].map(json.dumps)
    ).to_csv(OUT / "cyto_profiles.csv", index=False)

    print("\n=== POWER ===")
    print(power_df.to_string(index=False))
    print("\n=== KLD (median per method) ===")
    print(kld_df.groupby("method")["kl"].median().round(4).to_string())
    make_kld_plot(kld_df, OUT / "cyto_kld.png")
    print("cyto_symmetric done.")


if __name__ == "__main__":
    main()
