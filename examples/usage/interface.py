#%%
# Download a model if not present already:
from grappa.utils.run_utils import load_model

url = 'https://github.com/LeifSeute/test_torchhub/releases/download/test_release/protein_test_11302023.pth'

model = load_model(url)
# %%
# build a grappa molecule from bonds, charges and improper indices:
from grappa.data import Molecule

# we build a toy-molecule, methanoic acid, with one improper torsion (just as toy example, the parameters will not be physically sensible)
atoms = [1, 2, 3, 4, 5]
bonds = [(1,2), (2,3), (1,4), (1,5)]
impropers = [(1,2,4,5)]
partial_charges = [0.0, -0.1, -0.1, 0.1, 0.1]
atomic_numbers = [6, 8, 8, 1, 1]

mol = Molecule(atoms=atoms, bonds=bonds, impropers=impropers, partial_charges=partial_charges, atomic_numbers=atomic_numbers)

# %%
from grappa import Grappa

# build a grappa model that handles the ML pipeline
grappa = Grappa(model, device='cpu')

# predict parameters (grappa.data.Parameters) for the molecule
params = grappa.predict(mol)

# %%
# the params object now contains the parameters for the bonds, angles, propers and impropers in the units specified in grappa.units

print(f'bond ids:\n{params.bonds}\n')
print(f'bond k: \n{params.bond_k}\n')
print(f'bond eq: \n{params.bond_eq}\n')

print(f'proper ids:\n{params.propers}\n')
print(f'proper k: \n{params.proper_ks}\n')
print(f'proper phases: \n{params.proper_phases}\n')
# %%
