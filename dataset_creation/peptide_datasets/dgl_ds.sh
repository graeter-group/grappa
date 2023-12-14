#!/bin/bash

# stop upon error
# set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

source_path="$SCRIPT_DIR/../../data/grappa_datasets"
target_path="$SCRIPT_DIR/../../data/dgl_datasets"


# List of dataset names
datasets=("spice_dipeptide_amber99sbildn" "tripeptides_amber99sbildn" "tripeptides_openff210")

# Loop through each dataset name
for ds in "${datasets[@]}"; do
  python ../benchmark_datasets/to_dgl.py --source_path "$source_path/$ds" --target_path "$target_path/$ds"
done
