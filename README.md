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
### Processed data are shared via xx
```
seqscope_data_dict.pkl % the input that ELLA takes in seqscope
seqscope_df_registered_saved.pkl % registered expression data in seqscope
stereoseq_data_dict.pkl % the input that ELLA takes in stereoseq
stereoseq_df_registered_saved.pkl % registered expression data in stereoseq
seqfish_data_dict.pkl % the input that ELLA takes in seqfish
seqfish_df_registered_saved.pkl % registered expression data in seqfish
merfish_mouse_brai_data_dict.pkl % the input that ELLA takes in merfish_mouse_brain
merfish_mouse_brai_df_registered_saved.pkl % registered expression data in merfish_mouse_brain 
```
