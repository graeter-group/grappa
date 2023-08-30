"""
add the pubchem_more data to the pubchem data, deleting any molecules with the same name and saving a new dataset.
"""
#%%
from grappa.PDBData.PDBDataset import PDBDataset
from grappa.constants import DS_PATHS
from pathlib import Path
#%%
newpath = Path(DS_PATHS['spice_pubchem'])
name = newpath.name
#name = name.removesuffix("_filtered")

suffix = "_5"

# oldpath1 = str(newpath.parent) + "_old/" + name

oldpath1 = str(newpath.parent) + f"/{name}"
oldpath2 = str(newpath.parent) + f"_more{suffix}/" + name

#%%
ds1 = PDBDataset.load_npz(oldpath1)
ds2 = PDBDataset.load_npz(oldpath2)

# %%
add_mols = [mol for mol in ds2.mols if mol.name not in [m.name for m in ds1]]
ds2.mols = add_mols

# %%
ds3 = ds1 + ds2
# %%
ds3.save_npz(newpath, overwrite=True)
# %%
