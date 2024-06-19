---
title: MERFISH2
layout: default
parent: Reproducibility
nav_order: 5
---

### MERFISH Human Osteosarcoma Data Application

Here's the main code that we used to apply ELLA to the MERFISH human osteosarcoma data.
```
from ELLA.ELLA import model_beta, model_null, loss_ll, ELLA
ella_merfish2 = ELLA(dataset='merfish2')

# load data
ella_merfish2.load_data(data_path='input/merfish_data_dict.pkl')

# load resgistered cells
ella_merfish2.load_registered_cells()

# load prepared data for the NHPP fitting
ella_merfish2.load_nhpp_prepared()

# run nhpp fit
ella_merfish2.nhpp_fit()
```

