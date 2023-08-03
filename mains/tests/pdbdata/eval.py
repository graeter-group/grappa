#%%
from grappa.constants import DEFAULTBASEPATH
from grappa.PDBData.PDBDataset import PDBDataset

from grappa.ff import ForceField
from pathlib import Path

#%%

ds = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"spice/base", n_max=10)
ds.filter_confs()
# %%
ff = ForceField.from_tag("example")
ds.calc_ff_data(ff)
# %%
ds.evaluate(plotpath="plots", by_element=True, by_residue=True, suffix="")
# %%
from openmm.app import ForceField as OpenMMFF
ff = OpenMMFF("amber99sbildn.xml", "tip3p.xml")
ds.calc_ff_data(ff, suffix="amber99")
# %%
ds.evaluate(plotpath="plots", by_element=True, by_residue=True, suffix="", compare_name="Amber99sbildn", compare_suffix="amber99")
# %%
