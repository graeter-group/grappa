# create the spice dataset with amber99sbildn assuming grappa.constants.SPICEPATH and grappa.constants.DEFAULTBASEPATH are set correctly:
# the first two commands must only be run once

# python preprocess_spice.py # preprocess the spice file: filter dipeptides
# python spice.py # create unparametrised PDBDataset
python make_graphs.py --ds_name spice/base -ff amber99sbildn.xml -o --max_energy 65 --max_force 200