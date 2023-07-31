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
