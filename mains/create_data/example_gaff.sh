# create the spice dataset with gaff-2.11 assuming grappa.constants.SPICEPATH and grappa.constants.DEFAULTBASEPATH are set correctly:
# the first two commands must only be run once# the first two commands must only be run once

# load dataset
python spice.py --smiles
python make_graphs.py -off --ds_name spice_openff/base -ff gaff-2.11 -o