## Usage

Check out the [tutorial pages](https://jadexq.github.io/ELLA/) for demos and documentations to get started with ELLA.

### Installation
Prerequisites
- https://python-poetry.org/docs/#installation

```bash
git clone https://github.com/jadexq/ELLA.git
conda create -n ella python=3.9
conda activate ella
poetry install
```

### Commands

1. Simulate data
    ```bash
    python -m ella.data.simulate_data
    ```
    It will generate a `simulated_data.json` in the current directory
2. Train a model with a recipe, e.g. `configs/debug.yaml`
    ```bash
    ella-train --config-name debug
    ```
    It will save all outputs in `${log.save_dir}`
    (stopping rules is current hard coded!)
3. Open tensorboard
    ```bash
    tensorboard --logdir lightning_logs/debug/gene_0-kernel_-1
    ```
4. Estimate
   ```bash
   ella-estimate -d lightning_logs/debug -p "gene_0-kernel_.*" -b 10 -o path/to/out.json
   ```


### Data Preparation

1. Prepare data from pickle
   ```bash
   python -m ella.data.prepare_data -i your_dir/data.pkl -o prepared_data
   ```
2. Visualize data
   ```bash
   ella-visualize -d . -c '2102_686' -g 'Alb'
   ```

## Repo Structure
```
./ELLA/
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
