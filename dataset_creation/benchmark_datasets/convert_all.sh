#!/bin/bash
# convert the espaloma datasets to an openff-independent format

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

espaloma_ds_path="$SCRIPT_DIR/../../data/esp_data"
target_path="$SCRIPT_DIR/../../data/datasets"

# espaloma removed duplicates from their actual dataset and merged them to a new 'duplicate' dataset. we reverse this step and d the train-test splitting based on the unique smiles
python unmerge_duplicates.py --duplpath "$espaloma_ds_path/duplicated-isomeric-smiles-merge" --targetpath "$espaloma_ds_path"

# List of dataset names
datasets=("rna-nucleoside" "gen2" "spice-des-monomers" "spice-dipeptide" "rna-diverse" "gen2-torsion" "pepconf-dlc" "protein-torsion" "rna-trinucleotide" "spice-pubchem")
datasets=()

# Loop through each dataset name
for ds in "${datasets[@]}"; do
  python to_npz.py --dspath "$espaloma_ds_path/$ds" --targetpath "$target_path/$ds"
done
