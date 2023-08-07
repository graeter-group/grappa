#%%
from pathlib import Path
from grappa.PDBData.PDBDataset import PDBDataset


#%%
ds_path = Path(__file__).parent.parent.parent/"generate_data"/"data"/"pep100"

ds = PDBDataset.from_pdbs(ds_path, n_max=None, energy_name="openmm_energies.npy", force_name="openmm_forces.npy", xyz_name="positions.npy")
# %%
from openmm.app import ForceField as OpenMMFF
from grappa.ff import ForceField
ff = ForceField.from_tag("example")
amber = OpenMMFF("amber99sbildn.xml", "tip3p.xml")
ds.calc_ff_data(amber, suffix="amber99")
#%%
ds.calc_ff_data(ff)
# %%
ds.evaluate(plotpath="pep100_plots", by_element=True, by_residue=True, suffix="", compare_name="Amber99sbildn", compare_suffix="amber99", refname="Amber99sbildn", name="Grappa")
# %%

# now compare this with dipeptide amber energies:
ds_path = Path(__file__).parent.parent.parent/"generate_data"/"data"/"ref_dipeptides"
ds = PDBDataset.from_pdbs(ds_path, n_max=None, energy_name="openmm_energies.npy", force_name="openmm_forces.npy", xyz_name="positions.npy")
# %%
ds.calc_ff_data(amber, suffix="amber99")
#%%
ds.calc_ff_data(ff)
# %%
ds.evaluate(plotpath="dipep_plots", by_element=True, by_residue=True, suffix="", refname="Amber99sbildn", name="Grappa")
# %%
