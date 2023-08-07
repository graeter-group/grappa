#%%
from grappa.PDBData.PDBDataset import PDBDataset
from grappa.constants import DEFAULTBASEPATH
from pathlib import Path

#%%
ds = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"monomers"/"charge_default_ff_gaff-2_11_filtered", n_max=None)
# %%
ds.evaluate(plotpath="monomer_plots", by_element=True, by_residue=False, suffix="_total_ref", radicals=False)
# %%
ds = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"qca_spice"/"charge_default_ff_gaff-2_11_filtered", n_max=None)
# %%
ds.evaluate(plotpath="qca_s[ice_plots", by_element=True, by_residue=False, suffix="_total_ref", radicals=False)
# %%
