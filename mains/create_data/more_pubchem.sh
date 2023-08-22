# openff is required!
# replace --dipeppath by the path of your hdf5 file
# create the spice dataset with gaff-2.11 assuming grappa.constants.SPICEPATH and grappa.constants.DEFAULTBASEPATH are set correctly:
# the first two commands must only be run once

N_MAX=1000
set -e
source /hits/fast/mbm/seutelf/.bashrc_user
conda activate next_try # must be an env with grappa and openff installed
# load dataset

# obtain list of strings that contain the names of molecules that are already in the dataset
# list is printed by python script

names=$(python name_list.py spice_pubchem)
python spice.py --smiles --dipeppath /hits/fast/mbm/seutelf/data/datasets/pubchem_spice.hdf5 --name pubchem_more --n_max $N_MAX --skip_names $names --seed 1
python make_graphs.py -off -ff gaff-2.11 -o --ds_name pubchem_more/base --max_energy 65 --max_force 200