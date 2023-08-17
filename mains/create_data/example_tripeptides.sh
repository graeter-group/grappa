# create the spice dataset with amber99sbildn assuming grappa.constants.SPICEPATH and grappa.constants.DEFAULTBASEPATH are set correctly:
# the first two commands must only be run once
set -e
python tripeptides.py
python make_graphs.py --ds_name tripeptides/base -o --collagen --max_energy 100 --max_force 200