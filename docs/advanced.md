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

ELLA fits each gene independently and has no built-in parallelism, so a large panel (thousands of genes over hundreds of cells) run serially can take hours. Parallelizing rests on two facts:

- **Registration and data preparation depend only on the cells**, not on any individual gene, so `register_cells` + `nhpp_prepare` are run **once**.
- **The fit is per gene and deterministic** (the bounded-Newton solver has a unique global optimum, and a gene's result depends only on its own prepared data). So the gene set can be split into batches that fit independently and give **identical** results regardless of how they were batched.

The pattern: prepare once, save, then hand each worker only its batch of the prepared data. `subset_prepared(prepared, genes)` slices a prepared-data dict down to a gene subset, and `load_nhpp_prepared(prepared_dict=...)` loads that subset directly.

**1. Prepare once and save.**

```python
ella.register_cells()   # writes output/df_registered.pkl
ella.nhpp_prepare()     # writes output/df_nhpp_prepared.pkl
```

**2. Fit gene batches in parallel — give each worker only its batch.** Pin BLAS to one thread per worker (before importing numpy/ELLA), so many workers do not each spawn a thread pool and oversubscribe the cores.

```python
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from ELLA import ELLA, subset_prepared

full = pd.read_pickle('output/df_nhpp_prepared.pkl')     # load ONCE, in the parent
genes = full['gene_list_dict']['my_cell_type']
batches = [genes[i::n_jobs] for i in range(n_jobs)]      # or contiguous ranges

def fit_batch(sub):                                      # sub = this batch's prepared data
    ella = ELLA(dataset='my_run')
    ella.load_nhpp_prepared(prepared_dict=sub)
    ella.nhpp_fit(); ella.weighted_density_est(); ella.compute_pv()
    t = ella.type_list[0]
    return list(zip(ella.gene_list_dict[t], ella.pv_cauchy_tl[t]))   # raw (pre-FDR) p

with ProcessPoolExecutor(max_workers=n_jobs) as ex:
    results = list(ex.map(fit_batch, (subset_prepared(full, b) for b in batches)))
```

Slicing in the parent and passing only the subset means each gene's prepared data is sent to exactly one worker — peak memory scales with the batch size, not `panel × workers`, and no worker reloads the full panel.

**3. Or, for a job array / independent tasks**, pre-split the prepared data into per-shard files (there is no shared parent to slice), and have each task load only its file:

```python
# split step (run once)
full = pd.read_pickle('output/df_nhpp_prepared.pkl')
for i, b in enumerate(batches):
    pd.to_pickle(subset_prepared(full, b), f'output/prepared_{i:03d}.pkl')

# each array task
ella.load_nhpp_prepared(prepared_path=f'output/prepared_{task_id:03d}.pkl')
ella.nhpp_fit(); ella.weighted_density_est(); ella.compute_pv()
```

**4. Match the workers to the machine.** Run on a compute node, not a login node. Set `n_jobs` near the number of available cores; going above the number of genes just leaves workers idle.

**5. Apply the FDR correction once, at the end.** Collect the raw per-gene p-values from all batches into one array and apply the multiple-testing correction across the full gene set (not per batch).
