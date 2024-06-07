#!/bin/bash

set -e # exit on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

source_path="$SCRIPT_DIR/../../../../hartmaec/workdir/new_dataset/dataset_clean"
target_path="$SCRIPT_DIR/../../data/datasets"


# List of dataset names
datasets=("AA_radical")

target_ds_names=("AA_bondbreak_rad_amber99sbildn")

# Loop through each dataset name
for i in "${!datasets[@]}"; do
    ds="${datasets[$i]}"
    target_ds_name="${target_ds_names[$i]}"
    forcefield="${forcefields[$i]}"
    forcefield_type="${forcefield_types[$i]}"
    echo "Processing $ds"


    python ds_from_dirs.py --source_path "$source_path/$ds" --target_path "$target_path/$target_ds_name" --skip J_zetaCD1rad J_zetaCD2rad J_zetaCArad J_zetaBrad J_zetaCE2rad J_zetaOH2rad


done
