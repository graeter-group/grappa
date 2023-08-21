# create the spice dataset with amber99sbildn assuming grappa.constants.SPICEPATH and grappa.constants.DEFAULTBASEPATH are set correctly:
# the first two commands must only be run once
set -e
python large_peptides.py
python make_graphs.py --ds_name large_peptides/base -o --collagen --max_energy 1000 --max_force 200