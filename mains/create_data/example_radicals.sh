# create the spice dataset with amber99sbildn assuming grappa.constants.SPICEPATH and grappa.constants.DEFAULTBASEPATH are set correctly:
# the first two commands must only be run once
set -e
python radicals.py # create unparametrised PDBDataset
python make_graphs.py --ds_name radical_AAs/base -o -r --collagen -c heavy --max_energy 100 --max_force 200

python make_graphs.py --ds_name radical_dipeptides/base -o -r --collagen -c heavy --max_energy 100 --max_force 200