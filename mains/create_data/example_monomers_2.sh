# create the spice dataset with gaff-2.11 assuming grappa.constants.SPICEPATH and grappa.constants.DEFAULTBASEPATH are set correctly:
# the first two commands must only be run once# the first two commands must only be run once
set -e
# load dataset
source /hits/fast/mbm/seutelf/.bashrc_user
conda activate next_try # must be an env with grappa and openff installed
python spice.py --smiles --dipeppath /hits/fast/mbm/seutelf/compare_espaloma/download-qca-datasets/openff-default/Dataset/spice-des-monomers/SPICE-DES-MONOMERS-OPENFF-DEFAULT.hdf5 --name monomers2
python make_graphs.py -off -ff gaff-2.11 -o --ds_name monomers2/base --max_energy 65 --max_force 200