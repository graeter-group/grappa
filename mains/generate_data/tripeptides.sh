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
suffix=$1
n_molecules=$2
n_states_per_molecule=$3

# Append suffix to folder name
folder="data/pep3_$suffix"

# Original folder
orig_folder="data/pep3"


conda activate pepgen
python generate_pdbs.py --n_max "$n_molecules" -l 3 --folder "$folder"

conda activate psi4
python generate_states.py "$folder"/ -n "$n_states_per_molecule" --temperature 300 --plot

python single_points.py "$folder"/ --skip_errs

python validate_qm.py "$folder"/

# Check if original folder exists, if not create it
if [ ! -d "$orig_folder" ]; then
  mkdir -p "$orig_folder"
fi

# Copy all subfolders from the folder with _arg to the original folder
rsync -a "$folder"/ "$orig_folder"/