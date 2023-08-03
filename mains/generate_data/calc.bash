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

conda activate psi4_cascade

# Get arguments
suffix=${1:-1}
memory=${2:-8}
num_threads=${3:-1}

# Original folder
orig_folder="data/pep3"

# Append suffix to folder name
folder="$orig_folder"_"$suffix"

python single_points.py "$folder"/ --skip_errs --memory "$memory" --num_threads "$num_threads"

python validate_qm.py "$folder"/

# Check if original folder exists, if not create it
if [ ! -d "$orig_folder" ]; then
  mkdir -p "$orig_folder"
fi

# Copy all subfolders from the folder with _arg to the original folder
rsync -a "$folder"/ "$orig_folder"/

rm *.clean