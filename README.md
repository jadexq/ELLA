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
Complete Demo
- [complete_demo_data.pkl](sha256:aac2f957f98358b1622ddafa2a27b5482101ed0ce7fca3400916a06f37f92ec7)

Seq-Scope
- [data_health_PC](sha256:1e72aaa2ed4563bc0c122d3bb84975b5930903d1552936c6faafea3b72f9b84b)
- [data_health_PP](sha256:11b31911fd0fe9c24ad53dbe0194fd60781c0fbf12f82a8ef03a24d4c679cc45)
- [data_TD_PC](sha256:9619e72cf5ff8f4d10e125f7001b4d3e90deb87755db0c2618bdf553d96d46bb)
- [data_TD_PP](sha256:ee7f963d92f3fa5f5c7d57b76acb8ceab845db89867d35291c05e12b955a403a)

Stereo-seq
- [data_Myoblasts.pkl](sha256:03880a2626f6348a8638f68c46da2c29741dd2cd546eab0becc5039a521a26d3)
- [data_Cardiomyocytes.pkl](sha256:8b6a7373c1e68037421c26ac552c7e7c81eb7ca4f350a22c8ba1ea2d81b646f4)
  
SeqFISH+
- [data_Fibroblast.pkl](sha256:91fd8a86cd275f0fbe3f4dcd136c4f0eec9b37c4a8199e33eb7ae9f77ac44e29)

Merfish mouse brain
- [data_EX.pkl](sha256:256bc31125997b1a3ab3959ece95202d26e54d916f6c85fc156dc10591a67939)
- [data_IN.pkl](sha256:f58173aa24400b221d1fee6b5b8a71b9c5827318c9ead040f632639501d9a9cd)
- [data_Astr.pkl](sha256:4df33d2e58f8d3620d89790aa2d464979ce4da89efe4234be441e0ed6ab74471)
- [data_Oligo.pkl](sha256:cdcc63585c2c99fff009a524ef6f2fc4fc0d0586aaf855c81b9c229f2b517414)
