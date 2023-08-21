#%%
loadpath = "/hits/fast/mbm/seutelf/grappa/mains/generate_data/data/pep100"

storepath = "/hits/fast/mbm/seutelf/data/datasets/PDBDatasets/large_peptides/base"
MAX_ENERGY = 1e4
MAX_FORCE = 400

from grappa.PDBData.PDBDataset import PDBDataset
# %%
ds = PDBDataset.from_pdbs(path=loadpath, energy_name="openmm_energies.npy", force_name="openmm_forces.npy", allow_incomplete=True, n_max=100)
ds.filter_confs(max_energy=MAX_ENERGY, max_force=MAX_FORCE, reference=False)
ds.save_npz(storepath, overwrite=True)
# %%
