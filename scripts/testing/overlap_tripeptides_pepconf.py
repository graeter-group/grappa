#%%
from grappa.data import Dataset
#%%
tripep = Dataset.from_tag("tripeptides_amber99sbildn")
pepconf = Dataset.from_tag("pepconf-dlc")

#%%
tripep_smiles = tripep.mol_ids
pepconf_smiles = pepconf.mol_ids

print(f'Overlap between pepconf and tripeptides: {len(set(tripep_smiles).intersection(set(pepconf_smiles)))} molecules')
# %%
