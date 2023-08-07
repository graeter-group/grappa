# the first two commands must only be run once
set -e

source /home/seutelf/.bashrc

conda activate pepgen

# python generate_pdbs.py --n_max 10 -l 100 --folder data/pep100

# python generate_states.py data/pep100/ -n 50 --temperature 300 --plot

python generate_pdbs.py --n_max 20 -l 2 --folder data/ref_dipeptides

python generate_states.py data/ref_dipeptides -n 50 --temperature 300