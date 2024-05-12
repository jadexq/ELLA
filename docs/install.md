---
title: Intall ELLA
layout: default
nav_order: 2
---

Let's get ELLA installed :)

Create a conda envirnment for ELLA:
```
conda create -n ELLA python=3.10
conda activate ELLA
conda install conda-forge::r-base
```

Get ELLA from github:
```
git clone git@github.com:jadexq/ELLA.git
```

Go to the folder `ELLA` and install:
```
pip install -r requirements.txt
pip install .
```

Now, you've got ELLA installed. Please test ELLA tested with the  minimum demo here.

