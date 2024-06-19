---
title: MERFISH1
layout: default
parent: Reproducibility
nav_order: 4
---

### MERFISH Adult Mouse Brain Data Application

<div style="margin: 0 auto; text-align: left;"> 
<img src="{{ site.baseurl }}/images/rp_merfish1.png" width="500" />
</div>

Here's the main code that we used to apply ELLA to the MERFISH adult mouse brain data. We conducted the analysis across cell types and gene groups in parallel to save memory consumption and computation time.

We first ran the cell registrition across cell types for every 100 cells in parallel and the results were saved under a `output` folder.
```
#!/usr/bin/env python3

import sys
j = sys.argv[1]
print(f'idx={j}')

from ELLA.ELLA import model_beta, model_null, loss_ll, ELLA
ella_merfish1 = ELLA(dataset='merfish1', max_ntanbin=15)

# load data
ella_merfish1.load_data(data_path=f'input/split_data_{j}.pkl')

# register cells
ella_merfish1.register_cells(outfile=f'output/df_registered_{j}.pkl')
```

We then further processed the registered data for the NHPP model fitting and the results were automatically saved under the `output` folder.
```
from ELLA.ELLA import model_beta, model_null, loss_ll, ELLA
ella_merfish1 = ELLA(dataset='merfish1', max_ntanbin=15)

# load data
ella_merfish1.load_data(data_path='input/merfish_mouse_brain_data_dict.pkl')

# load registered cells
import pickle
import pandas as pd
rgst_dict = {}
for j in range(26):
    print(j)
    path = f'output/df_registered_{j}.pkl'
    rgst_dict[j] = pd.read_pickle(path)['df_registered']
combined_df = pd.concat(rgst_dict.values(), ignore_index=True)    
registered_dict = {}    
registered_dict['df_registered'] = combined_df
ella_merfish.load_registered_cells(registered_dict=registered_dict)
path = 'output/df_registered_merged.pkl'
with open(path, 'wb') as f:
    pickle.dump(registered_dict, f)

# prepare data for NHPP fit (r, c0, n etc.)
# ella_merfish.nhpp_prepare() 
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
ella_merfish1 = ELLA(dataset='merfish1', max_ntanbin=15)

# load data
print(f'load data')
ella_merfish1.load_data(data_path='input/merfish_mouse_brain_data_dict.pkl')

# load registered cells
print(f'load registered cells') ella_merfish1.load_registered_cells(registered_path='output/df_registered_saved.pkl')

# load prepared data
print(f'load prepared data for NHPP fit')
ella_merfish1.load_nhpp_prepared(data_path='output/df_nhpp_prepared_saved.pkl')

# <<<<< the cell type of focus
t = cell_type
ella_merfish1.type_list = [t]
print(ella_merfish1.type_list)

# <<<<< the subset of genes of focus
gl_full = ella_merfish1.gene_list_dict[t]
ella_merfish1.gene_list_dict[t] = gl_full[gene_idx_begin:gene_idx_end]

# run nhpp fit
print(f'run nhpp fit')
ella_merfish1.nhpp_fit(outfile=f'output/nhpp_fit_results_t{t}_g{gene_idx_begin}_{gene_idx_end}.pkl', ig_start=gene_idx_begin)
```

Other scripts used for mRNA characteristic analysis and for plotting are shared in the [github repo] (https://github.com/jadexq/ELLA/tree/main/scripts/merfish1).
