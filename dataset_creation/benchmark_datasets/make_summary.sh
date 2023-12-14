#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

target_path="$SCRIPT_DIR/../../data/datasets"
summary_path="$SCRIPT_DIR/../summaries"


# List of dataset names
datasets=("rna-nucleoside" "spice-des-monomers" "spice-dipeptide" "rna-diverse" "gen2" "gen2-torsion" "pepconf-dlc" "protein-torsion" "rna-trinucleotide" "spice-pubchem")

# Loop through each dataset name
for ds in "${datasets[@]}"; do
  python summary.py --targetpath "$target_path/$ds" --summarypath "$summary_path/$ds"
done