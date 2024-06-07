#%%
from grappa.data import Dataset

spice = Dataset.from_tag("spice-dipeptide")
# %%
own = Dataset.from_tag("dipeptides-300K-amber99")
# %%
smiles_spice = spice.mol_ids
smiles_own = own.mol_ids

# overlap:
overlap = set(smiles_spice).intersection(set(smiles_own))
print(len(list(overlap)))

# -> we cannot train on both spice dipeptides and our dipeptides, because the smiles strings are not compatible apparently
# %%
num_confs = [g.nodes['g'].data['energy_qm'].shape[-1] for g in spice.graphs]
# %%
