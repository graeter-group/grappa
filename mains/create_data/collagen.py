#%%
loadpath = "/hits/basement/mbm/seutelf/grappa/mains/generate_data/make_collagen_ds/data"
storepath = "/hits/fast/mbm/seutelf/data/datasets/PDBDatasets/collagen/base"
MAX_ENERGY = 200
MAX_FORCE = 400

from grappa.PDBData.PDBDataset import PDBDataset
# %%
ds = PDBDataset.from_pdbs(path=loadpath, energy_name="psi4_energies.npy", force_name="psi4_forces.npy", allow_incomplete=True)

ds.filter_confs(max_energy=MAX_ENERGY, max_force=MAX_FORCE, reference=False)
ds.save_npz(storepath, overwrite=True)