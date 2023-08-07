#%%
from grappa.constants import DEFAULTBASEPATH
from grappa.PDBData.PDBDataset import PDBDataset

from grappa.ff import ForceField
from pathlib import Path
from grappa.ff_utils.classical_ff.collagen_utility import get_mod_amber99sbildn


#%%
ds = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"AA_opt_rad/base", n_max=None)
ds.filter_confs()
# %%
ff = ForceField.from_tag("radical_example")
ds.calc_ff_data(ff, collagen=True, allow_radicals=True)
# %%
ds.evaluate(plotpath="rad_plots", by_element=True, by_residue=True, suffix="", radicals=True)
# %%
col_ff = get_mod_amber99sbildn()
ds.calc_ff_data(col_ff, suffix="amber99", allow_radicals=True, collagen=True)
# %%
ds.evaluate(plotpath="rad_plots", by_element=True, by_residue=True, suffix="", compare_name="Amber99sbildn", compare_suffix="amber99", radicals=True)
# %%
mol = ds[0]
print(mol.sequence)
# %%
