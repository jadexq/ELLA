#!/bin/bash

set -e

start=0
end=49

args=()
for i in $(seq $start $end); do
    args+=("data.gene_idx=$i")
done

for arg in "${args[@]}"; do
  sbatch \
    -J ELLA2 \
    -p main \
    --exclude=mulan-mc[01-05] \
    -t 0-2 \
    -c 10 \
    --mem=10g \
    --wrap "ella-train --config-name debug_complete_demo $arg" \
    --output log/%x.%j.out
done

# --exclude=mulan-mc[01-03] \