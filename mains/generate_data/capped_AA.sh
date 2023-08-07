# run with arguments suffix, n_molecules, n_states_per_molecule

# will do the calculations in the subfolder with suffix and copy everyhing to the original folder at the end.

# install pepgen (https://github.com/hits-mbm-dev/pepgen) in a conda environment named pepgen
# install psi4, ase, openmm, numpy and matplotlib in an environment named psi4.
# then replace the lines below:

#!/bin/bash
set -e

# MODIFY THIS
###############################################
# source /home/seutelf/.bashrc
# source /hits/basement/mbm/seutelf/.bashrc_user
source /hits/fast/mbm/seutelf/.bashrc_user
###############################################

# Get arguments
n_molecules=${1:-1}
n_states_per_molecule=${2:-10}
memory=${3:-8}
num_threads=${4:-1}

# Original folder
orig_folder="data/pep1"

# create the folder if it does not exist:
mkdir -p data
mkdir -p "$orig_folder"

# Find smallest non-occurring positive integer suffix
suffix=$(python find_suffix.py data/pep1)


# Append suffix to folder name
folder="$orig_folder"/"$suffix"

mkdir -p "$folder"

echo "Folder: $folder"


conda activate pepgen_cascade
python generate_pdbs.py --n_max "$n_molecules" -l 1 --folder "$folder"

conda activate grappa_cascade
python generate_states.py "$folder"/ -n "$n_states_per_molecule" --temperature 300 --plot

conda activate psi4_cascade
python single_points.py "$folder"/ --skip_errs --memory "$memory" --num_threads "$num_threads"

python validate_qm.py "$folder"/

# Check if original folder exists, if not create it
if [ ! -d "$orig_folder" ]; then
  mkdir -p "$orig_folder"
fi

# # Copy all subfolders from the folder with _arg to the original folder
# rsync -a "$folder"/ "$orig_folder"/

rm *.clean