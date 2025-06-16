---
title: Install ELLA
layout: default
parent: ella_v1
nav_order: 2
---

## Installation

<br>
**Let's get ELLA installed :)**
<br>
<br>

Create a conda envirnment for ELLA:

```
conda create -n ELLA python=3.10
conda activate ELLA
```
Install R:
```
conda install conda-forge::r-base
```
(Alternative ways of installing R can be found on the [R website](https://www.r-project.org).)


Get ELLA from github:

```
git clone https://github.com/jadexq/ELLA.git
```

Go to the folder `ELLA` and install:

```
cd ./ELLA
pip install -r requirements.txt
pip install -e .
```

Congrats, you've got ELLA installed! (The installation should take less than 5min.) You can get ELLA tested using [A Minimum Demo]({{ site.baseurl }}/mini_demo.html).

You may also install Jupyter Notebook for running the demo scripts:

```
pip install notebook
pip install ipywidgets
```

ELLA has been tested on operating systems: macOS Ventura 13.0, macOS Monterey 12.4, Ubuntu 20.04.6 LTS, Windows 10.

