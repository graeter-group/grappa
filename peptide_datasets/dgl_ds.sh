#!/bin/bash

# stop upon error
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

source_path="$SCRIPT_DIR/../data/peptides/grappa_datasets"
target_path="$SCRIPT_DIR/../data/peptides/dgl_datasets"


# List of dataset names
datasets=("spice")

# Loop through each dataset name
for ds in "${datasets[@]}"; do
  python ../benchmark_datasets/to_dgl.py --source_path "$source_path/$ds" --target_path "$target_path/$ds"
done
