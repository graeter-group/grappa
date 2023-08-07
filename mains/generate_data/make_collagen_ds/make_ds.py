#%%
from grappa.PDBData.PDBDataset import PDBDataset
# %%
ds = PDBDataset.from_pdbs(path="data", energy_name="psi4_energies.npy", force_name="psi4_forces.npy")
