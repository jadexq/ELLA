---
title: SeqFISH+
layout: default
parent: Reproducibility
nav_order: 3
---

### SeqFISH+ Mouse Embryonic Fibroblast Data Application

<div style="margin: 0 auto; text-align: left;"> 
<img src="{{ site.baseurl }}/images/rp_seqfish.png" width="500" />
</div>

Here's the main code that we used to apply ELLA to the seqFISH+ mouse embryonic fibroblast data.
```
from ELLA.ELLA import model_beta, model_null, loss_ll, ELLA
ella_seqfish = ELLA(dataset='seqfish+')

# load data
ella_seqfish.load_data(data_path='input/seqfish_data_dict.pkl')

# load resgistered cells
ella_seqfish.load_registered_cells()

# load prepared data for the NHPP fitting
ella_seqfish.load_nhpp_prepared()

# run nhpp fit
ella_seqfish.nhpp_fit()
```

