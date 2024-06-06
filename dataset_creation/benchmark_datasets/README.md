The scripts in this folder enable the reproduction of the creation of our benchmark dataset (which is also available as direct download).

The dataset is created from the dataset published along with espaloma. The form of the data points in the espaloma datasets are espaloma graphs. We store datasets mre genral, simply as npz files containing positions, energies, gradients, atomic numbers, partial charges and smiles strings.

Reproduction of the dataset creation requires the following steps:

1. Download the zip files: `bash download_all.sh`
2. Unzip the files: `bash extract_all.sh`
3. Convert from espaloma graphs (with espaloma and openff dependencies) to npz files: `bash convert_all.sh` This step also involves the inclusion of duplicates that were removed from the original datasets for some reason. In grappa, we store all data points and keep track of a mol_id, which is used to identify duplicate molecules upon splitting the dataset.
4. Create npz files in a format required by grappa: `bash grappa_ds.sh`. This will calculate nonbonded contributions except for partial charges from the forcefield openff-2.0.0 unconstrained. (Uses the partial charges provided by espaloma, i.e. am1bcc-elf)
4.5 Re-create the peptide dataset with amberff99sbildn as ref forcefield instead of openff-2.0.0: `bash convert_peptides.sh`

The resulting dataset has the same numbers of molecules as the espaloma benchmark dataset as can be seen in analyse.py.
The number of conformations, however, are differing form those reported in the paper. Here, espaloma seems to apply a different procedure to remove/merge duplicates. Our procedure is to simply unmerge the duplicates back to their original dataset since our splitting procedure can handle this. The un-merging is done in unmerge_duplicates.py which is called by convert_all.sh.