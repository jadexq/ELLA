## Usage

### Get Started

Execute the following commands in the project root directory
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

1. Prepare data from pickle (num of sector is currently hard coded! num of cpus may hard coded=8!)
   ```bash
   python -m ella.data.prepare_data -i /net/mulan/home/jadewang/revision/real_data/merfish/Oligo/data_Oligo.pkl -o prepared_data
   ```
2. Visualize data
   ```bash
   ella-visualize -d . -c '2102_686' -g 'Alb'
   ```





