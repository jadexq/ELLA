# ELLA

## Quick Start
Check out the [tutorial pages](https://jadexq.github.io/ELLA/) for demos and documentations to get started with ELLA.

## Quick Installation
```
git clone git@github.com:jadexq/ELLA.git
pip install -r requirements.txt
pip install -e .
```
Please check out [installation](https://jadexq.github.io/ELLA/install.html) for details.

## Repo Structure
```
├── setup.py % project config
├── requirements.txt % dependencies
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

## Processed Data
Seq-Scope
- the input that ELLA takes: [seqscope_data_dict.pkl](https://github.com/jadexq/ELLA/releases/download/v0.0.1/seqscope_data_dict.pkl)
- the registered expression data: [seqscope_df_registered_saved.pkl](https://github.com/jadexq/ELLA/releases/download/v0.0.1/seqscope_df_registered_saved.pkl)
  
Stereo-seq
- the input that ELLA takes: [stereoseq_data_dict.pkl](https://github.com/jadexq/ELLA/releases/download/v0.0.1/stereoseq_data_dict.pkl)
- the registered expression data: [stereoseq_df_registered_saved.pkl](https://github.com/jadexq/ELLA/releases/download/v0.0.1/stereoseq_df_registered_saved.pkl)
  
SeqFISH+
- the input that ELLA takes: [seqfish_data_dict.pkl](https://github.com/jadexq/ELLA/releases/download/v0.0.1/seqfish_data_dict.pkl)
- the registered expression data: [seqfish_df_registered_saved.pkl](https://github.com/jadexq/ELLA/releases/download/v0.0.1/seqfish_df_registered_saved.pkl)

Merfish mouse brain
- the input that ELLA takes: [merfish_mouse_brain_data_dict.pkl](https://github.com/jadexq/ELLA/releases/download/v0.0.1/merfish_mouse_brain_data_dict.pkl)
- the registered expression data: [merfish_mouse_brain_df_registered_saved.pkl](https://github.com/jadexq/ELLA/releases/download/v0.0.1/merfish_mouse_brain_df_registered_saved.pkl)

