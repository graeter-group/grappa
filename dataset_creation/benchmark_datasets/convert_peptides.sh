#!/bin/bash
# create peptide datasets with amber99sbildn energies and force as reference

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

espaloma_ds_path="$SCRIPT_DIR/../../data/esp_data"
target_path="$SCRIPT_DIR/../../data/grappa_datasets" # due to with_amber flag, this will be stored as moldata directly

# python unmerge_duplicates.py --duplpath "$espaloma_ds_path/duplicated-isomeric-smiles-merge" --targetpath "$espaloma_ds_path"

# List of dataset names
datasets=("spice-dipeptide")


# exclude some residues because of some apparant bug of openmmforcefields for this
HID_FRAGMENT='[N]([H])[C](=[O])[C]([H])([H])[H])[N]1[H]'
ASP_FRAGMENT='[C]([H])([H])[H])[C]([H])([H])[C](=[O])[O-])'

# Loop through each dataset name
for ds in "${datasets[@]}"; do
  python to_npz.py --dspath "$espaloma_ds_path/$ds" --targetpath "$target_path/$ds""_amber99sbildn" --with_amber99 --exclude_pattern $HID_FRAGMENT $ASP_FRAGMENT
done
