#%%
from grappa.PDBData.PDBDataset import PDBDataset
from grappa.ff import ForceField
from grappa.constants import DS_PATHS
from pathlib import Path
#%%
# mpath = "../../../mains/runs/compare_radicals/versions/35_med_scaled_1/best_model.pt"
ds_type = "spice_monomers"
# %%
ds = PDBDataset.load_npz(DS_PATHS[ds_type])
#%%
ds.grad_hist()
#%%
ds.energy_hist()
# %%

ds_tr, ds_vl, ds_te = ds.split_by_names(split_names=Path(mpath).parent/"split.json")
#%%
ds.evaluate(name="Gaff", suffix="_total_ref", plotpath="plots/"+ds_type, by_element=True, by_residue=False)
# %%
