#%%
from grappa.data.Dataset import Dataset

dataset = Dataset.from_tag('spice-dipeptide')
# %%
smiles = dataset.mol_ids

# check if a given string is contained in any of these:
target = '[C]([H])([H])[C@@]([H])([C](=[O])[N]([H])[C]([H])([H])[H])[N]([H])[C](=[O]'
target = '([C](=[O])[N]([H])[C]([H])([H])[H])'
target = '[N]([H])[C](=[O])[C]([H])([H])[H])[N]1[H]' # hid fragment

counter = 0
for smile in smiles:
    if target in smile:
        counter += 1
counter
# %%
hid_smiles = '[H][C]1=[N][C]([H])=[C]([C]([H])([H])[C@@]([H])([C](=[O])[N]([H])[C]([H])([H])[H])[N]([H])[C](=[O])[C]([H])([H])[H])[N]1[H]'
