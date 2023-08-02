#%%
from grappa.PDBData.PDBDataset import PDBDataset
from grappa.constants import DEFAULTBASEPATH
from pathlib import Path
from openmm.app import ForceField
#%%

ds = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"spice/base", n_max=10)
ds2 = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"spice/base", n_max=20)

ds.parametrize(forcefield=ForceField("amber99sbildn.xml"))
ds2.parametrize(forcefield=ForceField("amber99sbildn.xml"))
# %%
glist = ds.to_dgl(forcefield=ForceField("amber99sbildn.xml"))
# %%
_, names = ds2.split()
names
#%%
_, new_names = ds.split(existing_split_names=names)
new_names
# %%
from grappa.PDBData.PDBDataset import SplittedDataset

data = SplittedDataset.create([ds, ds2], [0.8, 0.1, 0.1])

# %%
full_loader,_,_ = data.get_full_loaders()
# %%
for g in full_loader:
    print(g.nodes["n1"].data["xyz"].shape)
# %%
_,_,loader1 = data.get_loaders(1)

for g in loader1:
    print(g.nodes["n1"].data["xyz"].shape)
# %%
data.save("test")

data2 = SplittedDataset.load("test", [ds, ds2])
# %%
full_loader,_,_ = data2.get_full_loaders()
for g in full_loader:
    print(g.nodes["n1"].data["xyz"].shape)

# %%
data3 = SplittedDataset.load_from_names("test", [ds, ds2])
# %%
full_loader,_,_ = data3.get_full_loaders()
for g in full_loader:
    print(g.nodes["n1"].data["xyz"].shape)
# %%
