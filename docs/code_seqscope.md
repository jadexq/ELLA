---
title: Code for Seq-Scope
layout: default
nav_order: 5
---

Here's the code that we used to analyze the Seq-Scope mouse liver data.

```python
from EG_analysis.EG_analysis import model_beta, model_null, loss_ll, EG_analysis

# <<<<< choose one cell type
t = 'H1'
print(f'cell type {t}')

eg_seqscope = EG_analysis(
    dataset='seqscope', 
    adam_learning_rate_max=5e-2, 
    adam_learning_rate_min=5e-3, 
    adam_learning_rate_adjust=1e7,
    adam_delta_loss_max=1e-3, 
    adam_delta_loss_min=1e-5, 
    adam_delta_loss_adjust=1e8,
    adam_niter_loss_unchange=20,
    max_iter=5000,
    min_iter=100,
    max_ntanbin=25,
    ri_clamp_min=0.01,
    ri_clamp_max=1
)

# load data
eg_seqscope.load_data(data_path='input/seqscope_data_dict.pkl')

# ntanbin
print(eg_seqscope.ntanbin_dict)

# load saved resgistered cells
eg_seqscope.load_registered_cells()

# load saved resgistered cells
eg_seqscope.load_nhpp_prepared()

# the chosen cell type
eg_seqscope.type_list = [t]
# print type list
print(eg_seqscope.type_list)

# run nhpp fit
eg_seqscope.nhpp_fit_parallel(outfile = f'output/nhpp_fit_results_{t}.pkl') 
```

