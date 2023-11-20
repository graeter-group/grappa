#%%
from grappa.data import Dataset, GraphDataLoader
import torch
from pathlib import Path
from grappa.utils import dgl_utils

# %%
dspath = Path(__file__).parents[1]/'data'/"dgl_datasets"

ds = Dataset.load(dspath/'spice-des-monomers')
g = ds[0][0]
g.nodes['g'].data.keys()
#%%
# ds += Dataset.load(dspath/'spice-dipeptide')
# %%
loader = GraphDataLoader(ds, batch_size=100, shuffle=True, num_workers=1, pin_memory=True)
# %%
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
for g, dsname in loader.to(device):
    print(f'n_instances: {g.num_nodes("g")}, num atoms: {g.nodes["n1"].data["xyz"].shape[0]}, dsname: {dsname})')
    graphs = dgl_utils.unbatch(g)
    assert len(graphs) == len(dsname)
    assert device in str(g.nodes['n1'].data['xyz'].device)
    g.nodes["g"].data["test"] = torch.ones(g.num_nodes("g"), 1).to(device)

# %%
g = next(iter(loader))[0]
g.nodes['g'].data.keys()
#%%

# %%

dspath = Path(__file__).parents[1]/'data'/"grappa_datasets"/"rna-nucleoside"
#%%
