#%%
from grappa.data import Dataset, GraphDataLoader
from pathlib import Path

#%%
datapath = Path(__file__).parents[2]/'data'/"dgl_datasets"

ds = Dataset.load(datapath/'spice-des-monomers').slice(None, 200)
ds += Dataset.load(datapath/'spice-dipeptide').slice(None, 400)

ds.remove_uncommon_features()

loader = GraphDataLoader(ds, batch_size=20, shuffle=True, weights={'spice-des-monomers': 4, 'spice-dipeptide': 2})

#%%
dscounts = {k:0 for k in ['spice-des-monomers', 'spice-dipeptide']}
for g, dsnames in loader.to('cuda'):
    assert 'cuda' in str(g.nodes['g'].data['energy_ref'].device)
    for dsname in dsnames:
        dscounts[dsname] += 1
print(dscounts)
# %%
# PRINT THE ACTUAL RATIOS:
dsratios = dscounts.copy()
total_num = sum(dscounts.values())
for k in dscounts.keys():
    dsratios[k] /= total_num
# %%
print(dsratios)
# %%
