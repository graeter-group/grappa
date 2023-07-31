#!/bin/bash
source /home/seutelf/.bashrc

# install pepgen (https://github.com/hits-mbm-dev/pepgen) in a conda environment named pepgen
# install psi4, ase, openmm, numpy and matplotlib in an environment named psi4

conda activate pepgen
python generate_pdbs.py --n_max 10 -l 3 --folder data/pep3

conda activate psi4
python generate_states.py data/pep3/ -n 10 --temperature 300 --plot

# python single_points.py data/pep3/ --skip_errs

# python validate_qm.py data/pep3/