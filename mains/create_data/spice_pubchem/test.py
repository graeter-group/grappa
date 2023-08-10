#%%
from grappa.constants import SPICEPATH
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from grappa.PDBData.PDBDataset import PDBDataset
from grappa.PDBData.PDBMolecule import PDBMolecule


#%%
# load the dataset:
spice = h5py.File(SPICEPATH, "r")
# %%
keys = list(spice.keys())
subsets = set([spice[key]["subset"][0] for key in keys])
#%%
pubchem_keys = [key for key in keys[:10] if "pubchem" in str(spice[key]["subset"][0].lower())]
# %%
spice.close()
# %%
type(list(subsets)[0])
# %%
str(list(subsets)[0])
# %%
pubchem_keys
# %%
