#%%
from grappa.PDBData.PDBDataset import PDBDataset
from grappa.constants import DEFAULTBASEPATH
from pathlib import Path

#%%
ds = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"monomers"/"charge_default_ff_gaff-2_11", n_max=None)

# %%
ds2 = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"qca_spice"/"charge_default_ff_gaff-2_11", n_max=None)
#%%
ds3 = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"pubchem"/"charge_default_ff_gaff-2_11", n_max=None)


# %%
ds.evaluate(plotpath="monomer_plots_unfiltered", by_element=True, by_residue=False, suffix="_total_ref", radicals=False)
# %%
ds2.evaluate(plotpath="qca_spice_plots_unfiltered", by_element=True, by_residue=False, suffix="_total_ref", radicals=False)
#%%
ds3.evaluate(plotpath="pubchem_plots_unfiltered", by_element=True, by_residue=False, suffix="_total_ref", radicals=False)
# %%
ds.filter_confs(max_energy=65, max_force=200, reference=False)
ds2.filter_confs(max_energy=65, max_force=200, reference=False)
ds3.filter_confs(max_energy=65, max_force=200, reference=False)
#%%
ds.save_npz(Path(DEFAULTBASEPATH)/"monomers"/"charge_default_ff_gaff-2_11_filtered", overwrite=True)
ds2.save_npz(Path(DEFAULTBASEPATH)/"qca_spice"/"charge_default_ff_gaff-2_11_filtered", overwrite=True)
ds3.save_npz(Path(DEFAULTBASEPATH)/"pubchem"/"charge_default_ff_gaff-2_11_filtered", overwrite=True)

# %%
ds.evaluate(plotpath="monomer_plots", by_element=True, by_residue=False, suffix="_total_ref", radicals=False)
ds2.evaluate(plotpath="qca_spice_plots", by_element=True, by_residue=False, suffix="_total_ref", radicals=False)
ds3.evaluate(plotpath="pubchem_plots", by_element=True, by_residue=False, suffix="_total_ref", radicals=False)
#%%
