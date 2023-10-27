#!/bin/bash

target_path="/hits/fast/mbm/seutelf/data/datasets"
summary_path="/hits/fast/mbm/seutelf/espaloma_orig/summaries"


# List of dataset names
datasets=("rna-nucleoside" "spice-des-monomers" "spice-dipeptide" "rna-diverse" "gen2" "gen2-torsion" "pepconf-dlc" "protein-torsion" "rna-trinucleotide" "spice-pubchem")

# Loop through each dataset name
for ds in "${datasets[@]}"; do
  python summary.py --targetpath "$target_path/$ds" --summarypath "$summary_path/$ds"
done