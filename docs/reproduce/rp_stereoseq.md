---
title: Stereo-seq
layout: default
parent: Reproducibility
nav_order: 2
---

### Stereo-seq Mouse Embryo Data Application

<div style="margin: 0 auto; text-align: left;"> 
<img src="{{ site.baseurl }}/images/rp_stereoseq.png" width="500" />
</div>

Here's the main code that we used to apply ELLA to the Stereo-seq mouse embryo data. For each cell type, we conducted the analysis for every 100 genes in parallel to significantly save memory consumption and computation time.

We first ran the cell registrition for both cell types and the results were automatically saved under a `output` folder.
```
from ELLA.ELLA import model_beta, model_null, loss_ll, ELLA
ella_stereoseq = ELLA(dataset='stereoseq', max_ntanbin=4)

# load data
ella_stereoseq.load_data(data_path='input/stereoseq_data_sub_dict.pkl')

# register all cells
ella_stereoseq.register_cells()
```

We then further processed the data for the NHPP model fitting and the results were automatically saved under the `output` folder.
```
from ELLA.ELLA import model_beta, model_null, loss_ll, ELLA
ella_stereoseq = ELLA(dataset='stereoseq', max_ntanbin=4)

# load data
ella_stereoseq.load_data(data_path='input/stereoseq_data_sub_dict.pkl')

# load registered cells
ella_stereoseq.load_registered_cells(registered_path='output/df_registered_saved.pkl')

# prepare data for NHPP fit (r, c0, n etc.)
ella_stereoseq.nhpp_prepare() 
```

We next ran the NHPP fitting and again, the results would be saved under the `output` folder.
```
#!/usr/bin/env python3

import sys
cell_type = sys.argv[1]
gene_idx_begin = int(sys.argv[2])
gene_idx_end = int(sys.argv[3])
print(f'cell_type {cell_type} gene_idx_begin {gene_idx_begin} gene_idx_end {gene_idx_end}')

from ELLA.ELLA import model_beta, model_null, loss_ll, ELLA
ella_stereoseq = ELLA(dataset='stereoseq', max_ntanbin=4)

# load data
print(f'load data')
ella_stereoseq.load_data(data_path='input/stereoseq_data_sub_dict.pkl')

# load registered cells
print(f'load registered cells')
ella_stereoseq.load_registered_cells(registered_path='output/df_registered_saved.pkl')

# prepare data for NHPP fit (r, c0, n etc.)
print(f'prepare data for NHPP fit')
ella_stereoseq.load_nhpp_prepared(data_path='output/df_nhpp_prepared_saved.pkl')

# <<<<< the cell type of focus
t = cell_type
ella_stereoseq.type_list = [t]
print(ella_stereoseq.type_list)

# <<<<< choose a subset of genes of focus
gl_full = ella_stereoseq.gene_list_dict[t]
ella_stereoseq.gene_list_dict[t] = gl_full[gene_idx_begin:gene_idx_end]

# run nhpp fit
print(f'run nhpp fit')
ella_stereoseq.nhpp_fit(outfile=f'output/nhpp_fit_results_t{t}_g{gene_idx_begin}_{gene_idx_end}.pkl', ig_start=gene_idx_begin) 
```

Other scripts used for mRNA characteristic analysis and for plotting are shared in the [github repo](https://github.com/jadexq/ELLA/tree/main/scripts/stereoseq).