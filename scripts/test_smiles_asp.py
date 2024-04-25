#%%
from openff.toolkit.topology import Molecule

mol = Molecule.from_polymer_pdb('DD.pdb')

mol.to_smiles(isomeric=True)
#%%
from grappa.data.Dataset import Dataset

dataset = Dataset.from_tag('spice-dipeptide')
# %%
smiles = dataset.mol_ids

# check if a given string is contained in any of these:
target = '[H][N]([C](=[O])[C@@]([H])([N]([H])[C](=[O])[C@@]([H])([N]([H])[C](=[O])[C]([H])([H])[H])[C]([H])([H])[C](=[O])[O-])[C]([H])([H])[C](=[O])[O-])[C]([H])([H])[H]'

asp_transformed = '[C]([H])([H])[H])[C]([H])([H])[C](=[O])[O-])'

target = asp_transformed

counter = 0
for smile in smiles:
    if target in smile:
        counter += 1
counter
# %%

# %%
