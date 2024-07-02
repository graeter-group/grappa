#%%
"""
This is an extension to the create_dataset tutorial, where we use openmm to calculate the nonbonded contribution of a classical force field to a dataset of QM energies and forces. For molecules that cannot be parametrized by openmm, or if nonbonded energy contributions are already available, the dataset can be created without the need of openmm, using the MolData.from_arrays constructor.
"""

import numpy as np
from pathlib import Path

from grappa.data import MolData, Molecule

#%%
# first, create a grappa.data.Molecule object that represents connectivity and atomic numbers of the molecule as described in the molecule_creation.py example:

atoms = [1, 2, 3, 4, 5]
bonds = [(1, 2), (1, 3), (1, 4), (1, 5)]
impropers = []
partial_charges = [-0.4, 0.1, 0.1, 0.1, 0.1]
atomic_numbers = [6, 1, 1, 1, 1]

methane = Molecule(atoms=atoms, bonds=bonds, impropers=impropers, partial_charges=partial_charges, atomic_numbers=atomic_numbers)

#%%

def get_data(n_confs:int=10):
    # For creating the MolData object, one needs arrays of shape:
    # - xyz: (n_confs, n_atoms, 3)
    # - energy: (n_confs,)                  (QM energies)
    # - gradient: (n_confs, n_atoms, 3)     (negative QM forces)
    # - nonbonded_energy: (n_confs,)        (QM - your classical ff contribution)
    # - nonbonded_gradient: (n_confs, n_atoms, 3)
    # - mol_id: str                         (used as identifier for dataset splitting)

    # create some random data for the example:
    n_atoms = 5

    # The following arrays should be obtained by your data generation process:
    xyz = np.random.rand(n_confs, n_atoms, 3)
    energy = np.random.rand(n_confs)
    gradient = np.random.rand(n_confs, n_atoms, 3)
    nonbonded_energy = np.random.rand(n_confs)
    nonbonded_gradient = np.random.rand(n_confs, n_atoms, 3)

    ff_energy = {"reference_ff": {"nonbonded": nonbonded_energy}}
    ff_gradient = {"reference_ff": {"nonbonded": nonbonded_gradient}}

    moldata = MolData(molecule=methane, xyz=xyz, energy=energy, gradient=gradient, ff_energy=ff_energy, ff_gradient=ff_gradient, mol_id="some_unique_identifier_string")
    return moldata
# %%
from grappa.utils import get_data_path

# generate a dataset that only contains this molecule with 100 conformations:
train_confs = get_data(100)

# save the dataset:
train_tag = 'example_train_set'
train_confs.save(get_data_path() / 'datasets' / train_tag / '0.npz')

# %%
# same for a val and test set:
val_confs = get_data(20)
val_tag = 'example_val_set'
val_confs.save(get_data_path() / 'datasets' / val_tag / '0.npz')

test_confs = get_data(20)
test_tag = 'example_test_set'
test_confs.save(get_data_path() / 'datasets' / test_tag / '0.npz')

#%%
s = """
Now we can train and evaluate on these datasets by appending these tags to a config file as done in configs/example_dataset.yaml.
To train on these datasets, run 'python experiments/train.py data=example_dataset' from the root directory.
Stop the training run by pressing ctrl+c.
Find the checkpoint path at 'ckpt/grappa/baseline/<date>/...'.
Evaluate it by running 'python experiments/evaluate.py ckpt_path=<path_to_checkpoint>'.
"""
print(s)
# %%
