---
title: Advanced Usage
layout: default
nav_order: 6
---

ELLA has been tested on high-resolution spatial transcriptomics datasets across various platforms and technologies. It comes with a set of default argument values that can be customized as needed. The usage of these customizable arguments is introduced in this page :)

The full list of customizable arguments and their default choices and functions are listed in the table below

| Args | Type | Default | Fucntion |
|------|------|---------|----------|
| `dataset` | str | 'Untitled' | Name of the dataset, help to distinguish multiple runs |
| `beta_kernel_param_list` | list of lists | 22 lists | Shape parameters of the 22 beta kernel functions in NHPP model fitting |
| `adam_learning_rate_max` | float | 1e-2 | Max initial learning rate of Adam |
| `adam_learning_rate_min` | float | 1e-3 | Min initial learning rate of Adam |
| `adam_learning_rate_adjust` | float | 1e7 | Adam LR = loglikelihood value under the null divided by 1e-7 |
| `adam_delta_loss_max` | float | 1e-2 | Max delta loss for Adam early stopping |
| `adam_delta_loss_min` | float | 1e-5 | Min delta loss for Adam early stopping |
| `adam_delta_loss_adjust` | float | 1e8 | Delta loss = loglikelihood value under the null divided by 1e-8 |
| `adam_niter_loss_unchange` | int | 20 | Adam stops if loss decrease < delta loss for 20 iterations |
| `max_iter` | int | 5e3 | Max number of interations in Adam |
| `min_iter` | int | 1e2 | Min bumber of interations in Adam |
| `max_ntanbin` | int | 25 | Number of bins for computing relative positions |
| `ri_clamp_min` | float | 1e-2 | Min relative position |
| `ri_clamp_max` | float | 1.0 | Max relative position |

The default values can be costomized while instantiating the class, for example
```
ella_demo = EG_analysis(dataset='Demo', max_iter=3000)
```



