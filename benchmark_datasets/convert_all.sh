#!/bin/bash

espaloma_ds_path="/hits/fast/mbm/seutelf/esp_data"
target_path="/hits/fast/mbm/seutelf/data/datasets"


# List of dataset names
datasets=("rna-nucleoside" "spice-des-monomers" "spice-dipeptide" "rna-diverse" "gen2" "gen2-torsion" "pepconf-dlc" "protein-torsion" "rna-trinucleotide" "spice-pubchem")

# Loop through each dataset name
for ds in "${datasets[@]}"; do
  python to_npz.py --dspath "$espaloma_ds_path/$ds" --targetpath "$target_path/$ds"
done

# add duplicates: (not, the duplicates provided from espaloma are not complete, calcuate them in the split function instead)
# python add_duplicates.py --duplpath "$espaloma_ds_path/duplicated-isomeric-smiles-merge" --targetpath "$target_path"
python create_split.py --dspath "$target_path" --splitpath "$target_path/../splits" --check --create_duplicates

