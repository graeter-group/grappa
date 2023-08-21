#%%
from pathlib import Path
from grappa.PDBData.PDBDataset import PDBDataset
from grappa.constants import DS_PATHS

#%%
ds_path = DS_PATHS["large_peptides"]

ds = PDBDataset.load_npz(ds_path, n_max=1)
# %%
from openmm.app import ForceField as OpenMMFF
from grappa.ff import ForceField

ff = ForceField.from_tag("example")

ds.eval_params(plotpath="pep100_plots", ff_name="Grappa", ff=ff, fontname="Arial")

#%%

amber = OpenMMFF("amber99sbildn.xml", "tip3p.xml")
ds.calc_ff_data(amber, suffix="amber99")
#%%
ds.calc_ff_data(ff)
# %%
ds.evaluate(plotpath="pep100_plots", by_element=True, by_residue=True, suffix="", compare_name="Amber99sbildn", compare_suffix="amber99", refname="Amber99sbildn", name="Grappa")

#%%
# %%
ds.mols[0].graph_data['n2'].keys()
#%%

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
