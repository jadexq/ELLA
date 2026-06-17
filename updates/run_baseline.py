"""
ELLA v1 baseline run on the mini demo (4 genes x 5 cells, fibroblast).

v1 has its OWN data format and run mechanism, distinct from v2:
  - input: a single .pkl dict {types, cells, cells_all, genes, cell_seg,
    nucleus_seg, expr} with raster segmentations (NOT v2's prepared jsonl).
  - run: one `ELLA` class instance, all genes/cells in one process, via the
    6-step API (load_data -> register_cells -> nhpp_prepare -> nhpp_fit ->
    weighted_density_est -> compute_pv). No Lightning / Hydra / per-gene CLI.
  - depends on R (rpy2 `stats::p.adjust` BY) -> run under env `ella1` so the
    activation hook sets LD_LIBRARY_PATH for libR.

This establishes the pre-update reference. Future v1 changes diff against the
artifacts written here. v1's lam(r) is in native distance-from-nucleus units and
is NOT comparable to v2's disk-frame baseline; this is for v1-vs-future-v1 only.

Reproducibility: the Adam fit is deterministic, but register_cells (cell
subsampling for bin count) and compute_pv (random p for degenerate kernels) use
numpy RNG, so we seed numpy/random/torch.

Run from repo root, on a compute node:
    conda run --no-capture-output -n ella1 python -u updates/run_baseline.py
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]                      # competing_methods/ELLA1
# ELLA/ is a namespace dir (no __init__.py) and setup.py find_packages() exposes
# nothing, so `import ELLA` only resolves when REPO is on sys.path. Add it
# explicitly so the import survives the chdir into _work below.
sys.path.insert(0, str(REPO))
DATA_PKL = REPO / "scripts" / "demo" / "mini_demo" / "input" / "mini_demo_data.pkl"
OUT = REPO / "updates" / "output" / "baseline"
WORK = OUT / "_work"                                            # hardcoded output/*.pkl land here
SEED = 42

# match the notebook's mini-demo constructor knobs
CTOR = dict(dataset="mini_demo", adam_learning_rate_min=1e-2, max_iter=1000)


def main() -> None:
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)  # ELLA hardcodes output/{pv_est,lam_est}.pkl relative to CWD

    from ELLA.ELLA import ELLA  # import after chdir; pulls in rpy2/R

    timing: dict = {}
    t_all = time.perf_counter()

    ella = ELLA(**CTOR)

    steps = [
        ("load_data", lambda: ella.load_data(data_path=str(DATA_PKL))),
        ("register_cells", ella.register_cells),
        ("nhpp_prepare", ella.nhpp_prepare),
        ("nhpp_fit", ella.nhpp_fit),
        ("weighted_density_est", ella.weighted_density_est),
        ("compute_pv", ella.compute_pv),
    ]
    for name, fn in steps:
        t0 = time.perf_counter()
        fn()
        timing[name] = round(time.perf_counter() - t0, 3)
        print(f"[baseline] {name}: {timing[name]:.3f}s", flush=True)

    timing["total_secs"] = round(time.perf_counter() - t_all, 3)

    # ---- collect per-type results ------------------------------------------
    lam_curves: dict = {}
    pvalues: dict = {}
    summary: dict = {"ctor": CTOR, "seed": SEED, "types": list(ella.type_list)}
    for t in ella.type_list:
        genes = list(ella.gene_list_dict[t])
        cells = list(ella.cell_list_dict[t])
        lam = np.asarray(ella.weighted_lam_est[t])
        pv_fdr = np.asarray(ella.pv_fdr_tl[t]).ravel()
        pv_cauchy = np.asarray(ella.pv_cauchy_tl[t]).ravel()
        best_kernel = np.asarray(ella.best_kernel_tl[t]).ravel()
        lam_curves[t] = {g: lam[i].tolist() for i, g in enumerate(genes)}
        pvalues[t] = {
            g: {
                "pv_fdr": float(pv_fdr[i]),
                "pv_cauchy": float(pv_cauchy[i]),
                "best_kernel": int(best_kernel[i]),
                "sig": bool(pv_fdr[i] <= ella.sig_cutoff),
            }
            for i, g in enumerate(genes)
        }
        summary[t] = {
            "genes": genes,
            "cells": cells,
            "n_genes": len(genes),
            "n_cells": len(cells),
            "n_sig": int(np.sum(pv_fdr <= ella.sig_cutoff)),
            "lam_npoints": int(lam.shape[1]) if lam.ndim == 2 else None,
        }

    (OUT / "timing.json").write_text(json.dumps(timing, indent=2))
    (OUT / "lam_curves.json").write_text(json.dumps(lam_curves, indent=2))
    (OUT / "pvalues.json").write_text(json.dumps(pvalues, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot(lam_curves, pvalues, OUT / "lam_curves.png")

    print("\n[baseline] DONE")
    print(f"  total: {timing['total_secs']:.1f}s  (fit: {timing['nhpp_fit']:.1f}s)")
    for t in ella.type_list:
        print(f"  {t}: {summary[t]['n_sig']}/{summary[t]['n_genes']} sig (FDR-BY<=0.05)")
        for g in summary[t]["genes"]:
            pv = pvalues[t][g]
            print(f"    {g:10s} pv_fdr={pv['pv_fdr']:.3e}  best_kernel={pv['best_kernel']}")
    print(f"\nwrote artifacts to {OUT}")


def _plot(lam_curves: dict, pvalues: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    types = list(lam_curves)
    genes0 = list(lam_curves[types[0]])
    nc = len(genes0)
    fig, axes = plt.subplots(len(types), nc, figsize=(2.2 * nc, 2.0 * len(types)),
                             squeeze=False)
    for ti, t in enumerate(types):
        for gi, g in enumerate(lam_curves[t]):
            ax = axes[ti][gi]
            curve = np.asarray(lam_curves[t][g])
            r = (np.arange(len(curve)) + 0.5) / len(curve)
            ax.plot(r, curve, lw=2, color="#c0362c")
            pv = pvalues[t][g]["pv_fdr"]
            ax.set_title(f"{g}\nFDR={pv:.1e}", fontsize=8)
            ax.set_xlabel("r"); ax.grid(alpha=0.3)
            if gi == 0:
                ax.set_ylabel(f"{t}\n$\\lambda(r)$", fontsize=8)
    fig.suptitle("ELLA v1 baseline: weighted lam(r), mini demo", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
