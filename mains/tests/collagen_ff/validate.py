#%%
from grappa.PDBData.PDBDataset import PDBDataset
from grappa.constants import DEFAULTBASEPATH
from pathlib import Path
from grappa.ff_utils.classical_ff.collagen_utility import get_old_collagen_ff, get_mod_amber99sbildn, get_amber99sbildn
old_ff = get_old_collagen_ff()

#%%
ds = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"AA_opt_nat"/"charge_default_col_ff_amber99sbildn_filtered", n_max=None)

# %%
ds.calc_ff_data(old_ff, suffix="_old", remove_errs=True, collagen=True)
ds.evaluate(plotpath="AA_opt_nat_plots", by_element=True, by_residue=True, suffix="_total_ref", compare_suffix="_old", name="mod_amber99sbildn", compare_name="from_rtp")



# %%
# now with radicals:
ds = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"AA_opt_rad"/"charge_heavy_col_ff_amber99sbildn_filtered", n_max=None)
#%%
ds.calc_ff_data(old_ff, suffix="_old", remove_errs=True, allow_radicals=True, collagen=True)

ds.evaluate(plotpath="AA_opt_rad_plots", by_element=True, by_residue=True, suffix="_total_ref", compare_suffix="_old", name="mod_amber99sbildn", compare_name="from_rtp", radicals=True)


# %%
# now with radicals:
ds = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"AA_scan_rad"/"charge_heavy_col_ff_amber99sbildn_filtered", n_max=None)
#%%
ds.calc_ff_data(old_ff, suffix="_old", remove_errs=True, allow_radicals=True, collagen=True)

ds.evaluate(plotpath="AA_scan_rad_plots", by_element=True, by_residue=True, suffix="_total_ref", compare_suffix="_old", name="mod_amber99sbildn", compare_name="from_rtp", radicals=True)
# %%

#%%
ds = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"AA_opt_nat"/"charge_default_col_ff_amber99sbildn_filtered", n_max=None)
# %%
ds.calc_ff_data(get_amber99sbildn(), suffix="_old", remove_errs=True, collagen=True)
ds.evaluate(plotpath="AA_opt_nat_amber", by_element=True, by_residue=True, suffix="_total_ref", compare_suffix="_old", name="mod_amber99sbildn", compare_name="amber")

# %%
