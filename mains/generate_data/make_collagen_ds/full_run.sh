#!/bin/bash

#SBATCH -o /hits/basement/mbm/seutelf/grappa/mains/generate_data/outfiles/info.out-%j
#SBATCH -t 24:00:00
#SBATCH --mem=16000
#SBATCH -n 4

set -e

#conda path:
CONDA_PREFIX="/hits/basement/mbm/seutelf/software/conda"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"


# Get arguments
sequence=${1:-"OA"}
n_states_per_molecule=${2:-50}
memory=${3:-15}
num_threads=${4:-4}

basefolder="/hits/basement/mbm/seutelf/grappa/mains/generate_data/make_collagen_ds"
pdb_folder="$basefolder"/data
folder="$pdb_folder"/"$sequence"

# create the folder if it does not exist:
mkdir -p "$pdb_folder"
mkdir -p "$folder"

echo "Folder: $folder"

cd /hits/basement/mbm/seutelf/grappa/mains/generate_data


echo "Generating data for sequence $sequence"
conda activate pepgen
python generate_pdbs2.py --folder "$folder"/"$sequence" --allow_collagen -s "$sequence"

echo "Generating states for sequence $sequence"
conda activate grappa_haswell
python generate_states.py "$folder" -n "$n_states_per_molecule" --temperature 300 --allow_collagen

echo "Calculating single points for sequence $sequence"
conda activate psi4
python single_points.py "$folder" --skip_errs --memory "$memory" --num_threads "$num_threads"

conda activate grappa_haswell
python validate_qm.py "$folder" --skip_errs --num_threads "$num_threads"