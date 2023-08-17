# create the spice dataset with gaff-2.11 assuming grappa.constants.SPICEPATH and grappa.constants.DEFAULTBASEPATH are set correctly:
# the first two commands must only be run once# the first two commands must only be run once
set -e
# load dataset
source /hits/fast/mbm/seutelf/.bashrc_user
conda activate next_try # must be an env with grappa and openff installed
python spice.py --smiles --name spice_openff_addf
python make_graphs.py -off --ds_name spice_openff_addf/base -ff gaff-2.11 -o --max_energy 65 --max_force 200