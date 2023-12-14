#!/bin/bash

# set -e # exit on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

source_path="$SCRIPT_DIR/../../../../hartmaec/workdir/new_dataset/dataset_clean"
target_path="$SCRIPT_DIR/../../data/grappa_datasets"
dglpath="$SCRIPT_DIR/../../data/dgl_datasets"


# List of dataset names
datasets=("AA_break/eq_average" "AA_break/eq_breakpoint" "AA_break/keep")

target_ds_names=("Capped_AA_break_average" "Capped_AA_break_breakpoint" "Capped_AA_break_keep")

# Define the string for README.md
readme_content=(
    'Fill this in.'
    'Fill this in.'
    'Fill this in.'
    )

# Loop through each dataset name
for i in "${!datasets[@]}"; do
    ds="${datasets[$i]}"
    target_ds_name="${target_ds_names[$i]}"
    forcefield="${forcefields[$i]}"
    forcefield_type="${forcefield_types[$i]}"
    echo "Processing $ds"
    python ds_from_dirs.py --source_path "$source_path/$ds" --target_path "$target_path/$target_ds_name"

    # Write to README.md in the target directory (create or overwrite)
    echo "${readme_content[$i]}" > "$target_path/$target_ds_name/README.md"

    # Convert to dgl dataset
    python ../benchmark_datasets/to_dgl.py --source_path "$target_path/$target_ds_name" --target_path "$dglpath/$target_ds_name"

done
