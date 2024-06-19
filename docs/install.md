---
title: Install ELLA
layout: default
nav_order: 2
---

## Installation

Let's get ELLA installed :)

- Create a conda envirnment for ELLA:

```
conda create -n ELLA python=3.10
conda activate ELLA
conda install conda-forge::r-base
```

- Get ELLA from github:

```
git clone git@github.com:jadexq/ELLA.git
```

- Go to the folder `ELLA` and install:

```
pip install -r requirements.txt
pip install -e .
```

Congrats, you've got ELLA installed! You can get ELLA tested using the minimum demo [here](https://jadexq.github.io/ELLA/demo.html).

[Optional] Install Jupyter Notebook if needed:

```
pip install notebook
pip install ipywidgets
```



