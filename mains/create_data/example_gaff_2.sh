# create the spice dataset with gaff-2.11 assuming grappa.constants.SPICEPATH and grappa.constants.DEFAULTBASEPATH are set correctly:
# the first two commands must only be run once# the first two commands must only be run once

# load dataset
python spice.py --smiles --dipeppath /hits/fast/mbm/seutelf/compare_espaloma/download-qca-datasets/openff-default/Dataset/spice-dipeptide/SPICE-DIPEPTIDE-OPENFF-DEFAULT.hdf5 --name qca_spice
python make_graphs.py -off --ds_name qca_spice/base -ff gaff-2.11 -o