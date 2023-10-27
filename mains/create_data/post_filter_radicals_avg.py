"""
Filter the radical datasets to eliminate molecules where the topology has not been matched correctly. These have large deviations in the classical ff forces.
"""

#%%
from grappa.constants import DS_PATHS
from grappa.PDBData.PDBDataset import PDBDataset

MAX_ENERGY = None
MAX_FORCE = 200

loadpath = DS_PATHS["radical_dipeptides_avg"]
# %%
ds = PDBDataset.load_npz(loadpath, n_max=None)

ds.filter_confs(max_energy=MAX_ENERGY, max_force=MAX_FORCE, deviation_from_ref=True)
ds.save_npz(loadpath, overwrite=True)
# %%
loadpath = DS_PATHS["radical_AAs_avg"]
# %%
ds = PDBDataset.load_npz(loadpath, n_max=None)

ds.filter_confs(max_energy=MAX_ENERGY, max_force=MAX_FORCE, deviation_from_ref=True)
ds.save_npz(loadpath, overwrite=True)
# %%
