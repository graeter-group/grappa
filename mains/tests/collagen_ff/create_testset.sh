# the first two commands must only be run once
set -e

source /hits/fast/mbm/seutelf/.bashrc_user

conda activate pepgen_cascade

python generate_pdbs.py -s AO AJ --folder data/ref_dipeptides

