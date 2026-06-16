"""
Fitting harness for the ELLA speedup work (baseline + optimized runs).

Runs the ELLA fitting pipeline on the 4-gene / 5-cell mini demo
(scripts/demo/mimi_demo) and records, into updates/output/<RUN_LABEL>/:

  - timing.json        per-gene + total wall-clock (the "before"/"after" number)
  - curves/gene_N_estimation_result.json  (ELLA's own estimate() output)
  - lam_curves.json    the estimated lam(r) per gene (lam_weighted)
  - lam_curves.png     4-panel plot of lam(r) for visual accuracy checks

Seeding: seed_everything is called before EACH kernel fit so every fit is
reproducible regardless of run order. This is the only deviation from the
stock pipeline; it makes baseline-vs-optimized comparison meaningful (the
stock model uses rsample() with no seed, so curves drift run to run).

Workflow: run with RUN_LABEL="baseline" before touching cox.py, then after the
speedup edits set RUN_LABEL="optimized" and re-run; output/baseline/ and
output/optimized/ sit side by side for compare.py to overlay.

Run from the ELLA repo root:
    conda run -n ella python -u updates/run_fit.py
"""
import json
import time
from pathlib import Path

import lightning as L
import numpy as np
from omegaconf import OmegaConf

from ella.cli.estimate import estimate
from ella.cli.train import train
from ella.options import ExperimentConfig, KernelParam

# ---- config --------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
CONFIG_YAML = REPO / "configs" / "mini_demo.yaml"
DATA_PATH = REPO / "scripts" / "demo" / "mimi_demo" / "prepared_data" / "training_data.jsonl"
GENES = [0, 1, 2, 3]
SEED = 42        # base seed; per-fit seed = SEED + 1000*gene_idx + (kernel_idx+1)
LR = 1e-3        # stock lr; stable now that cox.py forward is overflow-safe (issue #6 fix)
RUN_LABEL = "optimized"   # output/<RUN_LABEL>/ ; set "optimized" for the post-speedup run

OUT = Path(__file__).resolve().parent / "output" / RUN_LABEL
WORK = OUT / "_work"            # tensorboard logs + per-kernel result.json
CURVES = OUT / "curves"         # ELLA estimate() json per gene
OUT.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
CURVES.mkdir(parents=True, exist_ok=True)


def build_cfg(gene_idx: int, save_dir: Path, out_path: Path) -> ExperimentConfig:
    raw = OmegaConf.to_container(OmegaConf.load(CONFIG_YAML), resolve=False)
    est_n_bins = int(raw["estimation"]["n_bins"])
    return ExperimentConfig(
        log={"should_overwrite": True, "save_dir": str(save_dir)},
        data={"data_path": str(DATA_PATH), "gene_idx": gene_idx},
        kernel_params=[KernelParam(**kp) for kp in raw["kernel_params"]],
        model={
            "is_debug": True,
            "optimizer": {"lr": LR},
            "n_bins": int(raw["model"]["n_bins"]),
            "beta_init": float(raw["model"]["beta_init"]),
            "sigma02_init": float(raw["model"]["sigma02_init"]),
        },
        trainer={
            "max_epochs": int(raw["trainer"]["max_epochs"]),
            "enable_progress_bar": False,
        },
        estimation={
            "search_dir": str(save_dir),
            "log_dir_pattern": f"gene_{gene_idx}-kernel_",
            "output_path": str(out_path),
            "n_bins": est_n_bins,
        },
    )


def run_gene(gene_idx: int) -> dict:
    save_dir = WORK / f"gene_{gene_idx}"
    out_path = CURVES / f"gene_{gene_idx}_estimation_result.json"
    cfg = build_cfg(gene_idx, save_dir, out_path)

    log_dirs = []
    t0 = time.perf_counter()

    # null kernel first (matches train.main)
    L.seed_everything(SEED + 1000 * gene_idx + 0, workers=True)
    cfg_null = cfg.copy(deep=True)
    cfg_null.model.beta_init = 0.0
    cfg_null.model.beta_requires_grad = False
    log_dirs.append(train(cfg=cfg_null, kernel_idx=-1, kernel_param=KernelParam(a0=1.0, b0=1.0)))

    for k_idx, kp in enumerate(cfg.kernel_params):
        L.seed_everything(SEED + 1000 * gene_idx + (k_idx + 1), workers=True)
        log_dirs.append(train(cfg=cfg, kernel_idx=k_idx, kernel_param=kp))

    train_secs = time.perf_counter() - t0

    t1 = time.perf_counter()
    estimate(cfg=cfg.estimation, log_dirs=log_dirs)
    est_secs = time.perf_counter() - t1

    with out_path.open() as f:
        est = json.load(f)
        lam = est["lam_weighted"]
        gene_id = est.get("gene_id", str(gene_idx))

    return {"gene_idx": gene_idx, "gene_id": gene_id, "train_secs": train_secs,
            "estimate_secs": est_secs, "n_kernels": len(log_dirs), "lam": lam}


def main() -> None:
    print(f"baseline: {len(GENES)} genes, data={DATA_PATH.name}, seed base={SEED}")
    results = [run_gene(g) for g in GENES]

    timing = {
        "genes": [{k: r[k] for k in ("gene_idx", "train_secs", "estimate_secs", "n_kernels")}
                  for r in results],
        "total_secs": sum(r["train_secs"] + r["estimate_secs"] for r in results),
    }
    (OUT / "timing.json").write_text(json.dumps(timing, indent=2))

    lam_curves = {str(r["gene_idx"]): r["lam"] for r in results}
    (OUT / "lam_curves.json").write_text(json.dumps(lam_curves, indent=2))

    names = {str(r["gene_idx"]): r["gene_id"] for r in results}
    plot(lam_curves, names, OUT / "lam_curves.png")

    print("\n=== timing ===")
    for g in timing["genes"]:
        print(f"  gene {g['gene_idx']}: train {g['train_secs']:.1f}s + "
              f"estimate {g['estimate_secs']:.2f}s ({g['n_kernels']} kernels)")
    print(f"  TOTAL: {timing['total_secs']:.1f}s")
    print(f"\nwrote: {OUT}/timing.json, lam_curves.json, lam_curves.png")


def plot(lam_curves: dict, names: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    genes = sorted(lam_curves, key=int)
    fig, axes = plt.subplots(1, len(genes), figsize=(4 * len(genes), 3.2), squeeze=False)
    for ax, g in zip(axes[0], genes):
        lam = lam_curves[g]
        r = (np.arange(len(lam)) + 0.5) / len(lam)
        ax.plot(r, lam, lw=2)
        ax.set_title(f"{names.get(g, g)} (gene {g})")
        ax.set_xlabel("r (normalized radius)")
        ax.set_ylabel(r"$\lambda(r)$")
        ax.grid(alpha=0.3)
    fig.suptitle("ELLA baseline estimated lam(r) (seeded)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
