#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

espaloma_ds_path="$SCRIPT_DIR/../../data/esp_data"
target_path="$SCRIPT_DIR/../../data/datasets"


# List of dataset names
datasets=("rna-nucleoside" "spice-des-monomers" "spice-dipeptide" "rna-diverse" "gen2" "gen2-torsion" "pepconf-dlc" "protein-torsion" "rna-trinucleotide" "spice-pubchem")

# # Loop through each dataset name
# for ds in "${datasets[@]}"; do
#   python to_npz.py --dspath "$espaloma_ds_path/$ds" --targetpath "$target_path/$ds"
# done

python add_duplicates.py --duplpath "$espaloma_ds_path/duplicated-isomeric-smiles-merge" --targetpath "$target_path"