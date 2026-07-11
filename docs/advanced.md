---
title: Running ELLA
layout: default
nav_order: 6
---

## Running ELLA

The demos run ELLA inside a Jupyter notebook. ELLA is a plain Python library with no command-line interface, so "running ELLA" means calling its API from a script or a notebook. This page shows the standard run, plus a few adjustments for large panels (many cells and genes).

### The standard way

For small to moderate datasets (up to a few hundred cells and a few thousand genes), run the whole pipeline in one process, exactly as in [A Complete Demo]({{ site.baseurl }}/complete_demo.html):

```python
from ELLA.ELLA import ELLA

ella = ELLA(dataset='my_run')
ella.load_data(data_path='input/my_data.pkl')

# cell-level steps (geometry only, independent of genes)
ella.register_cells()
ella.nhpp_prepare()

# per-gene model fitting
ella.nhpp_fit()

# testing and estimation
ella.weighted_density_est()
ella.compute_pv()

# optional: cluster the significant genes by pattern
ella.pattern_clustering()
ella.pattern_labeling(K=5)
```

Results are stored on the object (`ella.pv_fdr_tl`, `ella.weighted_lam_est`, `ella.labels_dict`, ...); see [ELLA's Inputs and Outputs]({{ site.baseurl }}/inputs.html). Each stage also writes a pickle to `output/`.

### The large-panel way

ELLA fits each gene independently and has no built-in parallelism, so a large panel (thousands of genes over hundreds of cells) run serially can take hours. Four adjustments make it tractable. They all rest on one structural fact: **registration and data preparation depend only on the cells, while the expensive `nhpp_fit` is per gene and embarrassingly parallel across genes.**

**1. Register and prepare once, then reuse.** `register_cells` and `nhpp_prepare` do not depend on any individual gene, so run them a single time and reload the saved results instead of recomputing them:

```python
ella.load_registered_cells(registered_path='output/df_registered.pkl')
ella.load_nhpp_prepared(prepared_path='output/df_nhpp_prepared.pkl')
```

This removes the redundant re-registration of all cells that would otherwise happen in every parallel worker.

**2. Parallelize over genes.** Split the gene set into batches and fit each batch in its own process, then concatenate the per-gene results. Pin BLAS to a single thread per worker (before importing numpy/ELLA), so that many worker processes do not each spawn a BLAS thread pool and oversubscribe the cores:

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from concurrent.futures import ProcessPoolExecutor

def fit_gene_batch(gene_names):
    from ELLA.ELLA import ELLA
    ella = ELLA(dataset='my_run')
    ella.load_data(data_path='input/my_data.pkl')
    # reuse the one-time registration / preparation
    ella.load_registered_cells(registered_path='output/df_registered.pkl')
    ella.load_nhpp_prepared(prepared_path='output/df_nhpp_prepared.pkl')
    # restrict this worker to `gene_names`, then run
    #   nhpp_fit -> weighted_density_est -> compute_pv
    # and return each gene's raw (pre-FDR) p-value.
    ...

batches = [genes[i::n_jobs] for i in range(n_jobs)]   # or contiguous ranges
with ProcessPoolExecutor(max_workers=n_jobs) as ex:
    results = list(ex.map(fit_gene_batch, batches))
```

**3. Match the workers to the machine.** Run on a compute node, not a login node. Set `n_jobs` close to the number of available cores; going above the number of genes just leaves workers idle.

**4. Apply the FDR correction once, at the end.** Have the workers return the raw (pre-adjustment) per-gene p-values, collect them from all batches into one array, and apply the multiple-testing correction across the full gene set (not per batch).

Note: ELLA has no dedicated sharding API, and its internal gene-offset path can mis-index the output arrays. The reliable pattern is to have each worker fit its assigned genes explicitly, one gene at a time, and collect the p-values.
