#!/bin/bash
source /home/seutelf/.bashrc

# install pepgen (https://github.com/hits-mbm-dev/pepgen) in a conda environment named pepgen
# install psi4, ase, openmm, numpy and matplotlib in an environment named psi4

conda activate pepgen
python generate_pdbs.py --n_max 100 -l 1 --folder data/pep1

conda activate psi4
python generate_states.py data/pep1/ -n 10 --temperature 400

python single_points.py data/pep1/ --skip_errs

python validate_qm.py data/pep1/