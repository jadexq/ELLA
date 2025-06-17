#!/bin/bash

set -e

start=0
end=3

args=()
for i in $(seq $start $end); do
    args+=("data.gene_idx=$i")
done

for arg in "${args[@]}"; do
  sbatch \
    -J mini_demo \
    --wrap "ella-train --config-name mini_demo $arg" \
    --output log/%x.%j.out
done
