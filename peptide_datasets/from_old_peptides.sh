#!/bin/bash

# set -e # exit on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

source_path="$SCRIPT_DIR/../../old_data/datasets/PDBDatasets"
target_path="$SCRIPT_DIR/../data/grappa_datasets"


# List of dataset names
datasets=("spice" 'tripeptides')
datasets=("tripeptides")

target_ds_names=("spice_dipeptide_amber99sbildn" 'tripeptides_amber99sbildn')
target_ds_names=("tripeptides_amber99sbildn")

# Loop through each dataset name
for i in "${!datasets[@]}"; do
    ds="${datasets[$i]}"
    target_ds_name="${target_ds_names[$i]}"
    echo "Processing $ds"
    python ds_from_pdb.py --source_path "$source_path/$ds/charge_default_ff_amber99sbildn_filtered" --target_path "$target_path/$target_ds_name" --forcefield amber99sbildn.xml
done
