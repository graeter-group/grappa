#!/bin/bash

#SBATCH -o /hits/basement/mbm/seutelf/grappa/mains/generate_data/outfiles/info.out-%j
#SBATCH -t 24:00:00
#SBATCH --mem=42000
#SBATCH -n 4

set -e

#conda path:
CONDA_PREFIX="/hits/basement/mbm/seutelf/software/conda"

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"


# Get arguments
sequence=${1:-"OA"}
memory=${3:-38}
num_threads=${4:-4}

basefolder="/hits/basement/mbm/seutelf/grappa/mains/generate_data/make_collagen_ds"

pdb_folder="$basefolder"/data

folder="$pdb_folder"/"$sequence"


echo "Folder: $folder"

cd /hits/basement/mbm/seutelf/grappa/mains/generate_data



echo "Calculating single points for sequence $sequence"

conda activate psi4
python single_points.py "$folder" --skip_errs --memory "$memory" --num_threads "$num_threads"

conda activate grappa_haswell
python validate_qm.py "$folder" --skip_errs --num_threads "$num_threads"