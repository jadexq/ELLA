---
title: Code for Stero-seq
layout: default
nav_order: 6
---

Here's the code that we used to analyze the Stereo-seq mouse embryo E1S3 data.
```python
#!/usr/bin/env python3

import sys
cell_type = sys.argv[1]
gene_idx_begin = int(sys.argv[2])
gene_idx_end = int(sys.argv[3])
print(f'cell_type {cell_type} gene_idx_begin {gene_idx_begin} gene_idx_end {gene_idx_end}')

from EG_analysis.EG_analysis import model_beta, model_null, loss_ll, EG_analysis
eg_stereoseq = EG_analysis(
    dataset='stereoseq', 
    adam_learning_rate_max=1e-1, 
    adam_learning_rate_min=1e-2, 
    adam_learning_rate_adjust=1e7,
    adam_delta_loss_max=1e-3, 
    adam_delta_loss_min=1e-5, 
    adam_delta_loss_adjust=1e8,
    adam_niter_loss_unchange=20,
    max_iter=5000,
    min_iter=100,
    max_ntanbin=4,
    ri_clamp_min=0.01,
    ri_clamp_max=1
)

# load data
print(f'load data')
eg_stereoseq.load_data(data_path='input/stereoseq_data_dict.pkl')

# manually specify ntanbin
# ntanbin_dict = {}
# for t in eg_stereoseq.type_list:
#     ntanbin_dict[t] = 4 # <<<<<
# eg_stereoseq.specify_ntanbin(input_ntanbin_dict = ntanbin_dict)
print(eg_stereoseq.ntanbin_dict)

# load registered cells
print(f'load registered cells')
eg_stereoseq.load_registered_cells(registered_path='output/df_registered_saved.pkl')

# prepare data for NHPP fit (r, c0, n etc.)
print(f'prepare data for NHPP fit')
eg_stereoseq.load_nhpp_prepared(data_path='output/df_nhpp_prepared_saved.pkl')

# <<<<< choose one cell type
t = cell_type
eg_stereoseq.type_list = [t]
print(eg_stereoseq.type_list)

# <<<<< choose a subset of genes
gl_full = eg_stereoseq.gene_list_dict[t]
eg_stereoseq.gene_list_dict[t] = gl_full[gene_idx_begin:gene_idx_end]

# run nhpp fit
print(f'run nhpp fit')
eg_stereoseq.nhpp_fit(f'output/nhpp_fit_results_t{t}_g{gene_idx_begin}_{gene_idx_end}.pkl') 
```

