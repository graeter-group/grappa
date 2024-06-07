#!/bin/bash

# set -e # exit on first error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )" # dir in which this script lies

source_path="$SCRIPT_DIR/../../../old_data/datasets/PDBDatasets"
target_path="$SCRIPT_DIR/../../data/grappa_datasets"


# List of dataset names
# datasets=("AA_opt_rad" "AA_scan_rad" 'radical_AAs' 'radical_dipeptides')
datasets=('radical_dipeptides')

# target_ds_names=('Capped_AA_opt_rad' 'Capped_AA_scan_rad' 'Capped_AA_rad' 'dipeptide_rad')
target_ds_names=('dipeptide_rad')

# Define the string for README.md
# readme_content=(
#     'A Dataset of capped amino acids with a hydrogen detached (and thus being a radical). The states are opt trajectories.'
#     'A Dataset of capped amino acids with a hydrogen detached (and thus being a radical). The states are torsion scan trajectories.'
#     'A Dataset of capped amino acids with a hydrogen detached (and thus being a radical). The states sampled are from MD at 300K using a version of grappa that was trained on scans and opt trajectories of capped radical amino acids as forcefield.'
#     'A Dataset of capped dipeptides with a hydrogen detached (and thus being a radical). The states sampled are from MD at 300K using a version of grappa that was trained on scans and opt trajectories of capped radical amino acids as forcefield.'
#     )
readme_content=(
    'A Dataset of capped dipeptides with a hydrogen detached (and thus being a radical). The states sampled are from MD at 300K using a version of grappa that was trained on scans and opt trajectories of capped radical amino acids as forcefield.'
    )

# Loop through each dataset name
for i in "${!datasets[@]}"; do
    ds="${datasets[$i]}"
    target_ds_name="${target_ds_names[$i]}"
    forcefield="${forcefields[$i]}"
    forcefield_type="${forcefield_types[$i]}"
    echo "Processing $ds"
    python convert_radical_ds.py --source_path "$source_path/$ds/charge_heavy_col_ff_amber99sbildn_filtered" --target_path "$target_path/$target_ds_name"

    # Write to README.md in the target directory (create or overwrite)
    echo "${readme_content[$i]}" > "$target_path/$target_ds_name/README.md"
done
