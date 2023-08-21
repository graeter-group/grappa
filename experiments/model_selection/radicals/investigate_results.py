#%%
from grappa.PDBData.PDBDataset import PDBDataset
from grappa.ff import ForceField
from grappa.constants import DS_PATHS
from pathlib import Path
#%%
mpath = "../../../mains/runs/compare_radicals/versions/35_med_scaled_1/best_model.pt"
ds_type = "collagen"
#%%

ff = ForceField(model_path=mpath, allow_radicals=True)
# %%
ds = PDBDataset.load_npz(DS_PATHS[ds_type])
#%%
ds.grad_hist()
#%%
ds.energy_hist()
# %%
ds.calc_ff_data(ff)
# %%
ds_tr, ds_vl, ds_te = ds.split_by_names(split_names=Path(mpath).parent/"split.json")
#%%
ds_te.evaluate(suffix="", name="Grappa", compare_suffix="_total_ref", plotpath="plots/"+ds_type, compare_name="Amber99sbildn", by_element=True, by_residue=True, radicals=True)
# %%
