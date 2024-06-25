# ELLA
## Quick Start
Check out the [tutorial pages](https://jadexq.github.io/ELLA/) for detailed documentation and examples to get started with ELLA.

## Quick Installation
To install the latest version of the ELLA library from GitHub:
```
git clone git@github.com:jadexq/ELLA.git
pip install -r requirements.txt
pip install -e .
```
Please check out [installation](https://jadexq.github.io/ELLA/install.html) for details.

### This repo is orgnized as follows
```
├── setup.py %
├── requirements.txt %
├── ELLA % ELLA source code
│   └── ELLA.py
├── docs % source code of the tutorial website
│   ├── ...
├── processed_data % preprocessed data for each dataset (will be shared soon)
│   ├── seqscope
│   │   ├── seqscope_data_dict.pkl % the input that ELLA takes
│   │   └── df_registered_saved.pkl % registered expression data
│   ├── stereoseq
│   │   ├── stereoseq_data_dict.pkl
│   │   └── df_registered_saved.pkl
│   ├── seqfish
│   │   ├── seqfish_data_dict.pkl
│   │   └── df_registered_saved.pkl
│   └── merfish
│       ├── merfish_mouse_brain_data_dict.pkl
│       └── df_registered_saved.pkl
├── scripts
│   ├── demo % code and data for the minimum and complete demos
│   │   ├── mini_demo
│   │   └── complete_demo
│   ├── analysis % mRNA characteristic analysis code for each dataset
│   |   ├── merfish1
│   |   ├── seqfish
│   |   ├── seqscope
│   |   └── stereoseq
│   └── preprocessing % data preprocessing code for each data set
│       ├── merfish1
│       ├── seqfish
│       ├── seqscope
│       └── stereoseq
└── README.md

```
