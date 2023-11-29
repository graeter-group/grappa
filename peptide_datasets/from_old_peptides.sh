#!/bin/bash

# set -e # exit on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

source_path="$SCRIPT_DIR/../../old_data/datasets/PDBDatasets"
target_path="$SCRIPT_DIR/../data/peptides/grappa_datasets"


# List of dataset names
datasets=("spice")

# Loop through each dataset name
for ds in "${datasets[@]}"; do
  python ds_from_pdb.py --source_path "$source_path/$ds/charge_default_ff_amber99sbildn_filtered" --target_path "$target_path/$ds" --forcefield amber99sbildn.xml
done
