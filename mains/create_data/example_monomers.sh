# create the spice dataset with gaff-2.11 assuming grappa.constants.SPICEPATH and grappa.constants.DEFAULTBASEPATH are set correctly:
# the first two commands must only be run once# the first two commands must only be run once
set -e
# load dataset
python spice.py --smiles --dipeppath /hits/fast/mbm/seutelf/compare_espaloma/download-qca-datasets/openff-default/Dataset/spice-des-monomers/SPICE-DES-MONOMERS-OPENFF-DEFAULT.hdf5 --name monomers
python make_graphs.py -off -ff gaff-2.11 -o --ds_name monomers/base --max_energy 65 --max_force 200