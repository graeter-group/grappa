#!/bin/bash

# set -e # exit on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

source_path="$SCRIPT_DIR/../../../old_data/datasets/PDBDatasets"
target_path="$SCRIPT_DIR/../../data/grappa_datasets"


# List of dataset names
datasets=("spice" 'tripeptides' 'tripeptides')
datasets=("tripeptides")

target_ds_names=("spice_dipeptide_amber99sbildn" 'tripeptides_amber99sbildn' 'tripeptides_openff120')
target_ds_names=('tripeptides_openff120')

forcefields=("amber99sbildn.xml" "amber99sbildn.xml", "openff_unconstrained-1.2.0.offxml")
forcefields=("openff_unconstrained-1.2.0.offxml")

forcefield_types=("openmm" "openmm" "openff")
forcefield_types=("openff")

readme_content=(
    # 'A subset of the spice dipeptide dataset but with charges and thus nonbonded energies predicted from the amber99sbildn forcefield. Openff is used to infer a smiles string from the pdb file that had been inferred from the spice dataset.'
    # 'A dataset of randomly sampled tripeptides with states sampled from MD at 300K using the amber99sbildn forcefield. Charges and thus nonbonded energies are predicted from the amber99sbildn forcefield.'
    'A dataset of randomly sampled tripeptides with states sampled from MD at 300K using the amber99sbildn forcefield. Charges and thus nonbonded energies are predicted from the openff-1.2.0 unconstrained forcefield. The states are the same as in the tripeptides_amber99sbildn dataset.' 
    )

# Loop through each dataset name
for i in "${!datasets[@]}"; do
    ds="${datasets[$i]}"
    target_ds_name="${target_ds_names[$i]}"
    forcefield="${forcefields[$i]}"
    forcefield_type="${forcefield_types[$i]}"
    echo "Processing $ds with forcefield ${forcefields[$i]}"
    python ds_from_pdb.py --source_path "$source_path/$ds/charge_default_ff_amber99sbildn_filtered" --target_path "$target_path/$target_ds_name" --forcefield "$forcefield" --forcefield_type "$forcefield_type"
    
    # Write to README.md in the target directory (create or overwrite)
    echo "${readme_content[$i]}" > "$target_path/$target_ds_name/README.md"
done
