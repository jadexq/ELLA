---
title: Code for seqFISH+
layout: default
nav_order: 7
---

Here's the code that we used to analyze the seqFISH+ mouse embryonic fibroblast cell line data.

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import pi
import timeit 
import random
pd.options.mode.chained_assignment = None  # default='warn'!!
import sys

from EG_analysis.EG_analysis import model_beta, model_null, loss_ll, EG_analysis
eg_seqfish = EG_analysis(
    dataset='seq',
    adam_learning_rate_max=1e-1, 
    adam_learning_rate_min=1e-2, 
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
print(f'load data')
eg_seqfish.load_data(data_path='__input/seqfish_data_dict.pkl')
print(eg_seqfish.ntanbin_dict)

# load registered cells
print(f'load registered cells')
eg_seqfish.load_registered_cells(registered_path='__output/df_registered_saved_ntanbin10.pkl')

# load nhpp prepared cells
print(f'load prepared data for NHPP fit')
eg_seqfish.load_nhpp_prepared(data_path='__output/df_nhpp_prepared_saved_ntanbin10.pkl')

# run nhpp fit
print(f'run nhpp fit parallel')
eg_seqfish.nhpp_fit_parallel() 
```