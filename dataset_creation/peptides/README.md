Assumes that the amber99sbildn version has already been created using the grappa-data-creation repository.
The scripts provided
a) find the corresponding smiles string from the spice dataset and define it as mol_id for consistent train-val-test splitting
b) create the datasets with classical forcefield contributions from charmm36 and openff-1.2.0