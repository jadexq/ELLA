---
title: MERFISH
layout: default
parent: Reproducibility
nav_order: 4
---

### MERFISH Adult Mouse Brain Data Application

<div style="margin: 0 auto; text-align: left;"> 
<img src="{{ site.baseurl }}/images/rp_merfish1.png" width="500" />
</div>

For each cell type, we first preprocess the input data
``` python
python -m ella.data.prepare_data -i your_dir/data_EX.pkl -o prepared_data
```
Then run ELLA across genes in parallel for each cell type with a corresponding recipe (.yaml file)
```
bash run_merfish.sh
```

Other scripts used for mRNA characteristic analysis and for plotting are shared in the [github repo](https://github.com/jadexq/ELLA/tree/ella1/scripts/analysis/merfish).
