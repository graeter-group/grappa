#%%
from grappa.data.Dataset import Dataset
#%%
tags = ['spice-des-monomers', 'spice-dipeptide', 'hyp-dop_amber99sbildn']

datasets = [Dataset.from_tag(tag) for tag in tags]

#%%

monomer_dataset = datasets[0]

dgl_graph, _ = monomer_dataset[0]

print(dgl_graph.nodes['g'].data.keys())