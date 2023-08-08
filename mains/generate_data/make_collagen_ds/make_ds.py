#%%
from grappa.PDBData.PDBDataset import PDBDataset
# %%
ds = PDBDataset.from_pdbs(path="data", energy_name="psi4_energies.npy", force_name="psi4_forces.npy", allow_incomplete=True)

# %%
ds.parametrize()
# %%
ds.evaluate(plotpath="partial_plots", by_element=True, by_residue=True, suffix="_total_ref")
# %%
