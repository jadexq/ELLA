---
title: Install ELLA
layout: default
nav_order: 2
---

## Installation

<br>
**Let's get ELLA installed :)**
<br>
<br>

Requires Python ≥ 3.9. All dependencies are declared in `pyproject.toml` and installed automatically.

We recommend creating a conda environment for ELLA:

```
conda create -n ELLA "python>=3.9"
conda activate ELLA
```

**Option 1: install directly from GitHub**

```
pip install "git+https://github.com/jadexq/ELLA.git"
```

**Option 2: install from a local clone** (use `-e` for an editable/development install)

```
git clone https://github.com/jadexq/ELLA.git
cd ELLA
pip install -e .        # or: pip install .
```

Congrats, you've got ELLA installed! (The installation should take less than 5min.) You can get ELLA tested using [A Minimum Demo]({{ site.baseurl }}/mini_demo.html).

You may also install Jupyter Notebook for running the demo scripts:

```
pip install notebook
pip install ipywidgets
```

ELLA has been tested on operating systems: macOS Ventura 13.0, macOS Monterey 12.4, Ubuntu 20.04.6 LTS, Windows 10.

