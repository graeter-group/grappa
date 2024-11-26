#%%
'''
A minimal example for using Grappa without the pre-implemented interfaces (GROMACS/OpenMM).
'''

from grappa.data import Molecule
from grappa import Grappa

# define a methane molecular graph
atoms = [1, 2, 3, 4, 5]
atomic_numbers = [6, 1, 1, 1, 1]
bonds = [(1, 2), (1, 3), (1, 4), (1, 5)]

# impropers and partial charges have to be set by hand
impropers = []
partial_charges = [-0.4, 0.1, 0.1, 0.1, 0.1]

mol = Molecule(atoms=atoms, bonds=bonds, impropers=impropers, partial_charges=partial_charges, atomic_numbers=atomic_numbers)

#%%

# now we can load a pre-trained grappa model
model = Grappa.from_tag('latest')
# %%
# run the prediction
params = model.predict(mol)

# print the predicted bond parameters
print(f'Pred. eq distances: {params.bond_eq}')
print(f'Pred. force constants: {params.bond_k}')
print(f'Units: Directly derived by Angstroem, kcal/mol, radians, elementary charge')
# %%
