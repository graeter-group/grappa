#%%
from openff.toolkit.topology import Molecule

mol = Molecule.from_polymer_pdb('HIDHID.pdb')

mol.to_smiles(isomeric=True)
#%%
from grappa.data.dataset import Dataset

dataset = Dataset.from_tag('spice-dipeptide')
# %%
smiles = dataset.mol_ids

# check if a given string is contained in any of these:
target = '[C]([H])([H])[C@@]([H])([C](=[O])[N]([H])[C]([H])([H])[H])[N]([H])[C](=[O]'
target = '([C](=[O])[N]([H])[C]([H])([H])[H])'
target = '[N]([H])[C](=[O])[C]([H])([H])[H])[N]1[H]' # hid fragment
target = '[C]([H])([H])[C@@]([H])([C](=[O])[N]([H])[C]'

counter = 0
for smile in smiles:
    if target in smile:
        counter += 1
counter
# %%
hidhid_smiles = '[H][C]1=[N][C]([H])=[C]([C]([H])([H])[C@@]([H])([C](=[O])[N]([H])[C]([H])([H])[H])[N]([H])[C](=[O])[C@@]([H])([N]([H])[C](=[O])[C]([H])([H])[H])[C]([H])([H])[C]2=[C]([H])[N]=[C]([H])[N]2[H])[N]1[H]'