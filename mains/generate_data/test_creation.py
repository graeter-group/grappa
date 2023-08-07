#%%
from grappa.PDBData.PDBDataset import PDBDataset
# %%
ds = PDBDataset.from_pdbs(path="data/pep1", energy_name="psi4_energies.npy", force_name="psi4_forces.npy")
# %%
ds.parametrize()
# %%
ds.evaluate(suffix="_total_ref")
# %%
