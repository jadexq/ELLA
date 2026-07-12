"""benchmark.runner -- run ELLA (Newton or Adam) on one ELLA2D-format dataset.

Thin wrapper over bridge.ella1's machinery (dataset_to_v1_pickle, gene sharding,
per-gene seeding, register/prepare-ONCE) with two additions for the Newton-vs-Adam
benchmark:

  * the ELLA class is SELECTABLE -- source='newton' loads the shipped
    ELLA/ELLA.py; source='adam' loads the vendored benchmark/ella_adam/ELLA.py
    (a copy of HEAD with ONLY the alternative fit reverted to Adam). Everything
    else (ray-cast registration, nhpp_prepare, compute_pv, weighted_density_est)
    is identical, so a run difference isolates the optimizer.
  * run_dataset returns a wall-clock + peak-RSS record alongside the per-gene
    results (p-value + length-100 radial profile).

Import ONLY in the `ella1` conda env (the workers import torch + ELLA lazily).
"""
from __future__ import annotations

import importlib.util
import os
import socket
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BENCH_DIR = Path(__file__).resolve().parent
ELLA1_REPO = BENCH_DIR.parent                 # competing_methods/ELLA1
REPO_ROOT = ELLA1_REPO.parents[1]             # ella2d_v4
ADAM_ELLA_PATH = BENCH_DIR / "ella_adam" / "ELLA.py"

# bridge helpers (dataset conversion, sharding, strength). bridge.ella1's module
# body is torch/ELLA-free; torch/ELLA are imported lazily inside its workers.
sys.path.insert(0, str(REPO_ROOT / "competing_methods"))
from bridge.ella1 import (  # noqa: E402
    dataset_to_v1_pickle,
    _shard_indices,
    _strength_from_lam,
)

# public gene-slicer from the shipped ELLA package (torch-free -> safe to import in
# the parent before forking the fit pool). Used to hand each worker only its batch.
sys.path.insert(0, str(ELLA1_REPO))
from ELLA.ELLA import subset_prepared  # noqa: E402


def _load_ella(source: str):
    """Return the ELLA class for `source` ('newton' = shipped package, 'adam' =
    vendored benchmark copy). Called inside each worker process."""
    if source == "newton":
        if str(ELLA1_REPO) not in sys.path:
            sys.path.insert(0, str(ELLA1_REPO))
        from ELLA.ELLA import ELLA
        return ELLA
    if source == "adam":
        spec = importlib.util.spec_from_file_location("ella_adam_bench", str(ADAM_ELLA_PATH))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.ELLA
    raise ValueError(f"unknown ELLA source {source!r}")


# --------------------------------------------------------------------------- #
# peak-RSS sampler (this process + all worker children), mirrors eval_speed
# --------------------------------------------------------------------------- #
class PeakMem:
    """Sample max RSS over this process + all descendants on a background thread."""
    def __init__(self, interval: float = 0.1):
        import psutil
        self._psutil = psutil
        self._proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self._stop = False
        self._thr = None

    def _sample(self):
        while not self._stop:
            try:
                total = self._proc.memory_info().rss
                for ch in self._proc.children(recursive=True):
                    try:
                        total += ch.memory_info().rss
                    except self._psutil.Error:
                        pass
                if total > self.peak:
                    self.peak = total
            except self._psutil.Error:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self._thr = threading.Thread(target=self._sample, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        if self._thr is not None:
            self._thr.join(timeout=1.0)

    @property
    def peak_gb(self) -> float:
        return self.peak / 1e9


# --------------------------------------------------------------------------- #
# workers (module-level so ProcessPoolExecutor can pickle them)
# --------------------------------------------------------------------------- #
def _prepare_once(args):
    """load_data -> register_cells -> nhpp_prepare ONCE, return the prepared dict
    (standard ELLA schema: per-gene lists keyed by cell type) to the parent, which
    slices it per shard with subset_prepared. Runs in its own process so the parent
    stays torch-free before forking the fit pool. Also writes prepared.pkl (debug)."""
    pkl_path, source, ctor, setup_dir = args
    setup_dir = Path(setup_dir)
    setup_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(setup_dir)
    logf = open(setup_dir / "setup.log", "w")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = logf
    try:
        ELLA = _load_ella(source)
        ella = ELLA(dataset="bench", **ctor)
        ella.load_data(data_path=str(pkl_path))
        ella.register_cells()
        ella.nhpp_prepare()
        t = ella.type_list[0]
        prepared = {
            "type_list": list(ella.type_list),
            "gene_list_dict": {t: list(ella.gene_list_dict[t])},
            "cell_list_dict": ella.cell_list_dict,
            "cell_list_all": ella.cell_list_all,
            "r_tl": {t: list(ella.r_tl[t])},
            "c0_tl": {t: list(ella.c0_tl[t])},
            "c0_tl_homo": {t: list(ella.c0_tl_homo[t])},
            "n_tl": {t: list(ella.n_tl[t])},
            "n_tl_homo": {t: list(ella.n_tl_homo[t])},
        }
        pd.to_pickle(prepared, setup_dir / "prepared.pkl")
        return ("ok", prepared)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ("error", f"{type(e).__name__}: {e}")
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        logf.close()


def _fit_shard(args):
    """Fit ONE gene-batch: `sub` is this shard's prepared data (already sliced by
    the parent with subset_prepared), `gene_offset` is the batch's first global
    gene index. Loads the batch via the public load_nhpp_prepared(prepared_dict=),
    then fits its genes one at a time with per-gene seeding keyed to the GLOBAL
    index (seed_base + gene_offset + i) so results are identical regardless of how
    genes were batched. Returns per-gene dict(gene, pvalue, recovered_strength,
    lam_weighted)."""
    sub, source, gene_offset, ctor, seed_base, work_subdir = args
    work_subdir = Path(work_subdir)
    work_subdir.mkdir(parents=True, exist_ok=True)
    os.chdir(work_subdir)
    logf = open(work_subdir / "shard.log", "w")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = logf
    try:
        import numpy as _np
        import torch
        ELLA = _load_ella(source)

        ella = ELLA(dataset="bench", **ctor)
        ella.load_nhpp_prepared(prepared_dict=sub)   # public reload of this batch
        t = ella.type_list[0]
        shard_genes = list(ella.gene_list_dict[t])
        keys = ("r_tl", "c0_tl", "c0_tl_homo", "n_tl", "n_tl_homo")
        full = {k: list(getattr(ella, k)[t]) for k in keys}

        records = []
        for i, gene in enumerate(shard_genes):
            seed = int(seed_base + gene_offset + i)   # GLOBAL index -> batch-invariant
            _np.random.seed(seed)
            torch.manual_seed(seed)
            ella.gene_list_dict[t] = [gene]
            ella.ng_dict[t] = 1
            for k in keys:
                getattr(ella, k)[t] = [full[k][i]]
            ella.nhpp_fit()
            ella.weighted_density_est()
            ella.compute_pv()
            pc = float(np.asarray(ella.pv_cauchy_tl[t]).ravel()[0])
            lam = np.asarray(ella.weighted_lam_est[t])[0]
            records.append(dict(gene=gene, pvalue=pc,
                                recovered_strength=_strength_from_lam(lam),
                                lam_weighted=np.asarray(lam, float).tolist()))
        return records
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [{"gene": f"__ERROR__[offset={gene_offset}]", "pvalue": np.nan,
                 "recovered_strength": np.nan, "lam_weighted": None,
                 "_error": f"{type(e).__name__}: {e}"}]
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        logf.close()


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_dataset(dataset_dir, work_dir, *, source: str, n_jobs: int = 64,
                seed_base: int = 1000, max_genes: int | None = None,
                ctor: dict | None = None, log=print):
    """Run ELLA `source` on one dataset. Returns (df, meta):
      df   : DataFrame[gene, pvalue, recovered_strength, lam_weighted]
      meta : dict(method, cores, wall_s, peak_mem_gb, n_genes, n_cells, node)
    Registration + prep run ONCE (setup process); genes are sharded across n_jobs
    with per-gene seeding. Wall-clock + peak RSS wrap the whole run."""
    ctor = ctor or {}
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    pkl = work / "ella1_input.pkl"
    genes = dataset_to_v1_pickle(dataset_dir, pkl, log=log)
    n_cells = len(pd.read_pickle(pkl)["cells_all"])
    total = len(genes)
    n_fit = total if max_genes is None else min(max_genes, total)

    t0 = time.perf_counter()
    with PeakMem() as pm:
        setup_dir = work / "_setup"
        log(f"  [{source}] register+prepare {total} genes once ...")
        with ProcessPoolExecutor(max_workers=1) as ex:
            status, prepared = next(ex.map(_prepare_once, [(str(pkl), source, ctor, str(setup_dir))]))
        if status == "error":
            raise RuntimeError(f"ELLA {source} setup failed: {prepared} (see {setup_dir/'setup.log'})")
        t = prepared["type_list"][0]
        assert list(prepared["gene_list_dict"][t]) == genes, \
            "gene-order drift between conversion and register/prepare"

        # slice the prepared data per shard in the PARENT and hand each worker only
        # its batch (subset_prepared) -> each gene's data is transmitted once, and
        # no worker holds/reloads the full panel.
        shards = _shard_indices(n_fit, n_jobs)
        log(f"  [{source}] fitting {n_fit} genes across {len(shards)} shards (n_jobs={n_jobs}) ...")
        jobs = [(subset_prepared(prepared, genes[lo:hi]), source, lo, ctor, seed_base,
                 str(work / "shards" / f"{lo}_{hi}")) for lo, hi in shards]
        records = []
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs))) as ex:
            for rec in ex.map(_fit_shard, jobs):
                records.extend(rec)
    wall_s = time.perf_counter() - t0

    errs = [r for r in records if str(r["gene"]).startswith("__ERROR__")]
    for r in errs:
        log(f"  [{source}] SHARD ERROR {r['gene']}: {r.get('_error')}")
    df = pd.DataFrame([r for r in records if not str(r["gene"]).startswith("__ERROR__")])
    df = df.sort_values("gene").reset_index(drop=True)
    meta = dict(method=source, cores=n_jobs, wall_s=round(wall_s, 2),
                peak_mem_gb=round(pm.peak_gb, 3), n_genes=int(len(df)),
                n_cells=int(n_cells), node=socket.gethostname())
    log(f"  [{source}] done: {len(df)}/{n_fit} genes, wall={meta['wall_s']}s, "
        f"peak_mem={meta['peak_mem_gb']}GB, node={meta['node']}")
    return df, meta
