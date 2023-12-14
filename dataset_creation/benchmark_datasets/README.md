The scripts in this folder enable the reproduction of the creation of our benchmark dataset (which is also available as direct download).

The dataset is created from the dataset published along with espaloma. The form of the data points in the espaloma datasets are espaloma graphs. We store datasets mre genral, simply as npz files containing positions, energies, gradients, atomic numbers, partial charges and smiles strings.

Reproduction of the dataset creation requires the following steps:

1. Download the zip files: `bash download_all.sh`
2. Unzip the files: `bash extract_all.sh`
3. Convert from espaloma graphs (with espaloma and openff dependencies) to npz files: `python convert_all.py` This step also involves the inclusion of duplicates that were removed from the original datasets for some reason. In grappa, we store all data points and keep track of a mol_id, which is used to identify duplicate molecules upon splitting the dataset.
4. Create npz files in a format required by grappa: `bash grappa_ds.sh`. This will calculate nonbonded contributions except for partial charges from the forcefield openff-2.0.0 unconstrained. (Uses the partial charges provided by espaloma, i.e. am1bcc-elf)
5. Create dgl files that contain all graphs for a dataset: `bash dgl_ds.sh` This step also involves storing a mol_id and ds_name for each graph in json files as required by the grappa.data.Dataset class.

The resulting dataset has the same numbers of molecules as the espaloma benchmark dataset as can be seen in analyse.py.