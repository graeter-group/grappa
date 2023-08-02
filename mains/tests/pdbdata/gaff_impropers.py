#%%
from grappa.PDBData.PDBDataset import PDBDataset
from grappa.constants import DEFAULTBASEPATH
from pathlib import Path
#%%
ds = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"qca_spice/charge_default_ff_gaff-2_11")
# %%
impropers = 0
for mol in ds:
    if "n4_improper" in mol.to_dgl().ntypes:
        impropers+=1
# %%
impropers
# %%

ds = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"spice/charge_default_ff_amber99sbildn")
# %%
impropers = 0
for mol in ds:
    if "n4_improper" in mol.to_dgl().ntypes:
        impropers+=1
# %%
impropers
# %%
ds = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"monomers/charge_default_ff_gaff-2_11")
# %%
impropers = 0
for mol in ds:
    if "n4_improper" in mol.to_dgl().ntypes:
        impropers+=1
# %%
impropers
# %%
