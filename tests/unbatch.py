#%%
from grappa.data import Dataset

#%%
DSNAME = 'spice-dipeptide'
# Download a dataset if not present already:
dataset = Dataset.from_tag(DSNAME)

dataset = dataset.slice(0, 2)

# For more efficient data loading, we use a GraphDataLoader
from grappa.data import GraphDataLoader
from grappa.utils.dgl_utils import unbatch

loader = GraphDataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, conf_strategy='all')

from grappa.models.energy import Energy
from grappa.models.grappa import GrappaModel
import torch

model = torch.nn.Sequential(GrappaModel(), Energy())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# %%
for g, dsname in loader:
    loss = 0
    optimizer.zero_grad()
    g = model(g)
    # print(g.nodes['g'].data['energy_ref'])
    graphs = unbatch(g)
    for graph in graphs:
        energy_ref = graph.nodes['g'].data['energy_ref']
        energy = graph.nodes['g'].data['energy']

        loss += torch.nn.MSELoss()(energy, energy_ref)

        print(graph.nodes['g'].data['energy_ref'])
        print(graph.nodes['g'].data['energy'])


    loss.backward()
    optimizer.step()
# %%