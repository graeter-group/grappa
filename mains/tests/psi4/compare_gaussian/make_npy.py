#%%
from grappa.PDBData.PDBDataset import PDBDataset

from grappa.constants import DEFAULTBASEPATH
#%%
ds = PDBDataset.load_npz(DEFAULTBASEPATH+"/AA_scan_nat/base")
# %%
for m in ds:
    print(m.name)
# %%
i = 0
m = ds[i]
len(m)
# %%
n = 10
# choose n states of m randomly using numpy:
import numpy as np
np.random.seed(0)
indices = np.random.choice(len(m), n, replace=False)
# %%
energies = m.energies[indices]
forces = -m.gradients[indices]
positions = m.xyz[indices]
atomic_numbers = m.elements
# %%
# save the data:
import numpy as np

np.save("energies.npy", energies)
np.save("forces.npy", forces)
np.save("positions.npy", positions)
np.save("atomic_numbers.npy", atomic_numbers)
# %%

