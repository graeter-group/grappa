#%%
# shows that loading a dgl graphlist is much faster than loading a list of grappa graphs
from grappa.data import MolData
from pathlib import Path
from time import time
from random import random
#%%
datadir = Path(__file__).parents[2]/'data'/'grappa_datasets'
# %%
t0 = time()
graphs = []
split_encoding = []
dsname = 'spice-dipeptide'
for f in (datadir/dsname).iterdir():
    if f.suffix == '.npz':
        g = MolData.load(f)
        graphs.append(g.to_dgl())
        if random() < 0.5:
            split_encoding.append(1)
        else:
            split_encoding.append(0)

print(time()-t0)
# %%
len(graphs)
# %%
import dgl
import torch

# add second dimension to store several splits
labels = {"split_encoding":torch.tensor(split_encoding, dtype=torch.long).unsqueeze(dim=-1)}

dglpath = datadir.parent/'dgl_datasets'

dgl.save_graphs(str(dglpath/dsname)+".bin", g_list=graphs, labels=labels)
# %%
t0 = time()
graphs2, labels = dgl.load_graphs(str(dglpath/dsname)+".bin")
print(time()-t0)
# %%
