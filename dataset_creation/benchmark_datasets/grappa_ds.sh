#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

source_path="$SCRIPT_DIR/../../data/datasets"
target_path="$SCRIPT_DIR/../../data/grappa_datasets"


# List of dataset names
datasets=("rna-nucleoside" "gen2" "spice-des-monomers" "spice-dipeptide" "rna-diverse" "gen2-torsion" "pepconf-dlc" "protein-torsion" "rna-trinucleotide" "spice-pubchem")
datasets=("spice-dipeptidedes-monomers")

# Loop through each dataset name
for ds in "${datasets[@]}"; do
  python to_grappa.py --source_path "$source_path/$ds" --target_path "$target_path/$ds" --forcefield openff_unconstrained-2.0.0.offxml
done
