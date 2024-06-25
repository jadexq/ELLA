---
title: A Complete Demo
layout: default
nav_order: 4
---

## A Complete Demo


**Here's a demo introducing a relatively complete ELLA analysis pipline.** You probabaly have alreay installed ELLA, so let get start wtih downloading the input data for this demo from here. The data is a subset of the processed seqFISH+ embryonic fibroblast dataset. It is a .pkl file that stores a dictionnary of three data frames corresponding to the gene expression, cell segmentation, and nucleus segmentation (optional) with 20 cells and 50 genes. 

The scripts and data that will be used in this demo can be found [here](https://github.com/jadexq/ELLA/tree/main/scripts/demo/complete_demo).
They  should be orgnized as:
```
complete_demo/
├── input
│   └── complete_demo_data.pkl
├── output
│   ├── df_nhpp_prepared_saved.pkl
│   ├── df_registered_saved.pkl
│   ├── lam_est.pkl
│   ├── nhpp_fit_results_saved.pkl
│   └── pv_est.pkl
└── complete_demo.ipynb
```

1. Let's begin with initiating ELLA:
```python
# import ELLA
from ELLA.ELLA import model_beta, model_null, loss_ll, ELLA
ella_demo = ELLA(dataset='demo')
# load data
ella_demo.load_data(data_path='input/complete_demo_data.pkl')
```
2. Let's run the data pre-processing and model fitting steps:
```python
# register cells
ella_demo.register_cells()
# prepare data for model fitting
ella_demo.nhpp_prepare() 
# model fitting
ella_demo.nhpp_fit()
```
As this could take a couple of minutes, to save time,  let's instead download the saved results from here and put them into the `output` folder. ELLA can easily load saved results with:
```python
# load registered cells
ella_demo.load_registered_cells(path='output/df_registered_saved.pkl')
# load prepared data for model fitting
ella_demo.load_nhpp_prepared(path='output/df_nhpp_prepared_saved.pkl')
# load model fitting results
ella_demo.load_nhpp_fit_results(path='output/nhpp_fit_results_saved.pkl')
```
3. Let's then run the testing and estimation:
```python
# expression intensity estimation
ella_demo.weighted_density_est()
# likelihood ratio test
ella_demo.compute_pv()
```
4. We can now clustering the estimated (significant) expression intensities into clusters of patterns. We find the optimal number of kmeans clusters K with the ELBOW method where K is chosen as point where the distortion/inertia begins to decrease more slowly:
```python
ella_demo.pattern_clustering()
```
<div style="margin: 0 auto; text-align: center;"> 
<img src="{{ site.baseurl }}/images/demo2_elbow.png" width="500" />
</div>
Based on the plots, it seems 5 can be a proper choice, thus let's proceed with K=5 to obtain 5 pattern clusters:
```python
ella_demo.pattern_labeling(K=5)
```
Prints:
```bash
Pattern 1: 6 genes
Pattern 2: 14 genes
Pattern 3: 9 genes
Pattern 4: 10 genes
Pattern 5: 9 genes
```
<div style="margin: 0 auto; text-align: center;"> 
<img src="{{ site.baseurl }}/images/demo2_est.png" width="500" />
</div>
We can overlay genes in the same cluster in cells to have a more intuitive sense of the patterns.
<div style="margin: 0 auto; text-align: center;"> 
<img src="{{ site.baseurl }}/images/demo2_cells.png" width="600" />
</div>