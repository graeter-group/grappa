#%%
from grappa.ff_utils.SysWriter import SysWriter

import h5py
import numpy as np
#%%
from grappa.constants import SPICEPATH
spice = h5py.File(SPICEPATH, "r")
#%%
ala_smile = spice["ala-gly"]["smiles"][0]
#%%
w = SysWriter.from_smiles(ala_smile, ff="gaff-2.11")
w.use_residues = False
# %%
w.init_graph(with_parameters=True)
# %%
g = w.graph
# %%
g.nodes['n3'].data['eq_ref']
# %%
