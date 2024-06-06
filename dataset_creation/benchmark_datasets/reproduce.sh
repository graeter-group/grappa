# calling this script with every line commented out will recreate the benchmark datasets by downloading and converting to several formats the original datasets
# first from espaloma graph to npz representation of the exact data from espaloma
# then from npz to another npz format but in a more ordered, general way that can then be converted to grappas MolData class
# then from these MolData representation to a .bin file containing all molecules of the repective dataset as dgl graph.
# this requires an environment with grappa and openff installed!

# download the original datasets
bash download_all.sh
bash extract_all.sh

# convert the original datasets to npz
bash convert_all.sh

# convert the npz datasets to MolData
bash grappa_ds.sh