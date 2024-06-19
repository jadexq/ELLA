---
title: Seq-Scope
layout: default
parent: Reproducibility
nav_order: 1
---

### Seq-Scope Mouse Liver Data Application

<div style="margin: 0 auto; text-align: center;"> 
<img src="{{ site.baseurl }}/images/rp_seqscope.png" width="500" />
</div>

Here's the main code that we used to apply ELLA to the Seq-Scope mouse liver data. We conducted the analysis for each cell type in parallel to save memory consumption and computation time.

```
from ELLA.ELLA import model_beta, model_null, loss_ll, ELLA

# <<<<< manually specify one cell type
t = 'H1'
print(f'cell type {t}')

ella_seqscope = ELLA(dataset='seqscope')

# load data
ella_seqscope.load_data(data_path='input/seqscope_data_dict.pkl')

# load resgistered cells
ella_seqscope.load_registered_cells()

# load prepared data for the nhpp model fitting
ella_seqscope.load_nhpp_prepared()

# work on the chosen cell type
ella_seqscope.type_list = [t]
print(ella_seqscope.type_list)

# run nhpp fit
ella_seqscope.nhpp_fit(outfile = f'output/nhpp_fit_results_{t}.pkl') 
```

