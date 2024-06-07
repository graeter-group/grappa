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
# For creating the MolData object, one needs arrays of shape:
# - xyz: (n_confs, n_atoms, 3)
# - energy: (n_confs,)                  (QM energies)
# - gradient: (n_confs, n_atoms, 3)     (negative QM forces)
# - nonbonded_energy: (n_confs,)        (QM - your classical ff contribution)
# - nonbonded_gradient: (n_confs, n_atoms, 3)
# - mol_id: str                         (used as identifier for dataset splitting)

# create some random data for the example:
n_confs = 10
n_atoms = 5

xyz = np.random.rand(n_confs, n_atoms, 3)
energy = np.random.rand(n_confs)
gradient = np.random.rand(n_confs, n_atoms, 3)
nonbonded_energy = np.random.rand(n_confs)
nonbonded_gradient = np.random.rand(n_confs, n_atoms, 3)

moldata = MolData(molecule=methane, xyz=xyz, energy=energy, gradient=gradient, nonbonded_energy=nonbonded_energy, nonbonded_gradient=nonbonded_gradient, mol_id="dummy_smilestring")