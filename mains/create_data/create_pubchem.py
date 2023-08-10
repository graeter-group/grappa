"""
Sample random molecules from the pubchem subsets.
"""
#%%
SEED = 0
NUM_MOLECULES = None
#%%
from grappa.constants import SPICEPATH
import h5py
import numpy as np
from pathlib import Path

#%%

# load the full dataset:
spice = h5py.File(SPICEPATH, "r")
# get the keys of pubchem molecules:
pubchem_keys = [key for key in spice.keys() if "pubchem" in str(spice[key]["subset"][0].lower())]
print(f"found {len(pubchem_keys)} pubchem molecules")
#%%
# sample random molecules:
if NUM_MOLECULES is not None:
    np.random.seed(SEED)
    sampled_keys = np.random.choice(pubchem_keys, NUM_MOLECULES, replace=False).tolist()
else:
    sampled_keys = pubchem_keys
# %%
# write the data to a new, smaller hdf5 file:
pubchem_file = h5py.File(Path(SPICEPATH).parent/Path("pubchem_spice.hdf5"), "w")
for i, key in enumerate(sampled_keys):
    grp = pubchem_file.create_group(key)
    for subkey in spice[key].keys():
        pubchem_file[key][subkey] = spice[key][subkey][()]
#%%
pubchem_file.close()
spice.close()
# %%
