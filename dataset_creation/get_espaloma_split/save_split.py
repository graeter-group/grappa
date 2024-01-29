#%%
"""
Load all smiles, exclude test and val obtained from espaloma. than save as dict {'train':[train_smiles], 'val':[val_smiles], 'test':[test_smiles]} as espaloma_split.json
"""

DATASETS = ["gen2", "gen2-torsion", "pepconf-dlc", "protein-torsion", "spice-pubchem", "spice-dipeptide", "spice-des-monomers", "rna-diverse"]

from grappa.data import Dataset

full_ds = Dataset()
for ds_tag in DATASETS:
    print(f'loading {ds_tag}...')
    ds = Dataset.from_tag(ds_tag)
    full_ds += ds

#%%
import json

all_smiles = full_ds.mol_ids

with open('te_smiles.json', 'r') as f:
    te_smiles = json.load(f)
with open('vl_smiles.json', 'r') as f:
    vl_smiles = json.load(f)

all_smiles = set(all_smiles)

tr_smiles = (all_smiles - set(vl_smiles) ) - set(te_smiles)
# %%
print(f'found {len(tr_smiles)} unique train smiles')
print(f'found {len(set(vl_smiles))} unique val smiles')
print(f'found {len(set(te_smiles))} unique test smiles')

# %%
# %%
# test correctness
# assert that there are no overlaps:
    
assert len(set(tr_smiles) & set(vl_smiles)) == 0
assert len(set(tr_smiles) & set(te_smiles)) == 0
if len(set(vl_smiles) & set(te_smiles)) > 0:
    print(f'Train and val used in espaloma have overlapping smiles!\n{set(vl_smiles) & set(te_smiles)}')

print('removing the overlap from the val set...')

vl_smiles = list(set(vl_smiles) - set(te_smiles))

print(f'saving the split to espaloma_split.json')

with open('espaloma_split.json', 'w') as f:
    json.dump({'train':list(tr_smiles), 'val':vl_smiles, 'test':te_smiles}, f)
# %%
