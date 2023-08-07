# create the spice dataset with amber99sbildn assuming grappa.constants.SPICEPATH and grappa.constants.DEFAULTBASEPATH are set correctly:
# the first two commands must only be run once
# only continue if succesful:
set -e

# python scans_opt.py # create unparametrised PDBDataset
python make_graphs.py --ds_name AA_opt_nat/base AA_scan_nat/base --collagen -o --max_energy 65 --max_force 200
python make_graphs.py --ds_name AA_opt_rad/base AA_scan_rad/base --collagen -o -r --max_energy 65 --max_force 200 -c heavy