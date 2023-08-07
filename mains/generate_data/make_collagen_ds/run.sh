#!/bin/bash

#SBATCH -o /hits/basement/mbm/seutelf/grappa/mains/generate_data/outfiles/info.out-%j
#SBATCH -t 24:00:00
#SBATCH --mem=16000
#SBATCH -n 4


#conda path:
CONDA_PREFIX="/hits/basement/mbm/seutelf/software/conda"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"


# Get arguments
sequence=${1:-"OA"}
n_states_per_molecule=${2:-50}
memory=${3:-8}
num_threads=${4:-1}

basefolder="/hits/basement/mbm/seutelf/grappa/mains/generate_data/make_collagen_ds"
folder="$basefolder"/data/"$sequence"

# create the folder if it does not exist:
mkdir -p "$basefolder"/data
mkdir -p "$folder"

echo "Folder: $folder"

cd /hits/basement/mbm/seutelf/grappa/mains/generate_data


conda activate pepgen
python generate_pdbs.py --folder "$folder" --allow_collagen -s "$sequence"

conda activate grappa
python generate_states.py "$folder"/ -n "$n_states_per_molecule" --temperature 300 --plot --allow_collagen

conda activate psi4
python single_points.py "$folder"/ --skip_errs --memory "$memory" --num_threads "$num_threads"

