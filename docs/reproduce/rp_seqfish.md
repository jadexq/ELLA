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

For the fibroblast cell type, we first preprocess the input data
``` python
python -m ella.data.prepare_data -i your_dir/data_Fibroblast.pkl -o prepared_data
```
Then run ELLA across genes in parallel with a corresponding recipe (.yaml file)
```
bash run_seqfish.sh
```

Other scripts used for mRNA characteristic analysis and for plotting are shared in the [github repo](https://github.com/jadexq/ELLA/tree/ella1/scripts/analysis/seqfish).