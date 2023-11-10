#%%
from grappa.utils.dgl_utils import batch, unbatch

import dgl
import torch
import copy

from pathlib import Path
#%%

dglpath = Path(__file__).parents[1]/'data'/'dgl_datasets'
dsname = 'spice-dipeptide'
ds, _ = dgl.load_graphs(str(dglpath/dsname)+".bin")

#%%
num_confs = []
for g in ds:
    n = len(g.nodes['g'].data['energy_ref'][0])
    num_confs.append(n)
min(num_confs)

# make all graphs have the same number of conformations and batch them together
for g in ds:
    g.nodes['n1'].data['xyz'] = g.nodes['n1'].data['xyz'][:,:min(num_confs)]
    g.nodes['g'].data['energy_ref'] = g.nodes['g'].data['energy_ref'][:,:min(num_confs)]
    for feat in g.nodes['g'].data.keys():
        if 'energy' in feat:
            g.nodes['g'].data[feat] = g.nodes['g'].data[feat][:,:min(num_confs)]
    for feat in g.nodes['n1'].data.keys():
        if 'gradient' in feat:
            g.nodes['n1'].data[feat] = g.nodes['n1'].data[feat][:,:min(num_confs)]


batchsize = 3
orig_copy = copy.deepcopy(ds[:10])
batches = []
for i in range(0, len(ds), batchsize):
    batches.append(batch(ds[i:min(i+batchsize, len(ds))]))
# %%
g = batches[0]
# assert that every node takes part in a bond
assert torch.all(torch.arange(g.number_of_nodes('n1')) == torch.sort(g.nodes['n2'].data['idxs'].flatten().unique()).values)
#%%

# check that the unbatches graphs are the same as the original ones:
graphs = unbatch(batches[0])
assert len(graphs) == batchsize

for i, g in enumerate(graphs):
    for ntype in g.ntypes:
        for feat in g.nodes[ntype].data.keys():
            if not torch.all(g.nodes[ntype].data[feat] == orig_copy[i].nodes[ntype].data[feat]):
                raise ValueError(f"Graph {i}, node type {ntype}, feature {feat} does not match: \n{g.nodes[ntype].data[feat]}\nvs\n{ds[i].nodes[ntype].data[feat]}")
# %%
