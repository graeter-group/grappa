#%%
from grappa.data import Dataset, GraphDataLoader
import torch
from pathlib import Path
from grappa.utils import dgl_utils

# %%
dspath = Path(__file__).parents[1]/'data'/"dgl_datasets"

ds = Dataset.load(dspath/'spice-des-monomers')
ds += Dataset.load(dspath/'spice-dipeptide')
ds += Dataset.load(dspath/'protein-torsion')
# %%
split_dict = ds.calc_split_ids(partition=[(0.8,0.1,0.1), {'spice-des-monomers':(0, 0, 1), 'spice-dipeptide':(0.8, 0.1, 0.1), 'protein-torsion':(0.8, 0.1, 0.1)}], seed=20)

tr, vl, te = ds.split(train_ids=split_dict['train'], val_ids=split_dict['val'], test_ids=split_dict['test'], check_overlap=True)

loader = GraphDataLoader(vl, batch_size=100, shuffle=True, num_workers=1, pin_memory=True)
# %%
if torch.cuda.is_available():
    device = 'cuda'
for g, dsname in loader.to(device):
    print(f'n_instances: {g.num_nodes("g")}, num atoms: {g.nodes["n1"].data["xyz"].shape[0]}, dsname: {dsname})')
    graphs = dgl_utils.unbatch(g)
    assert len(graphs) == len(dsname)
    assert device in str(g.nodes['n1'].data['xyz'].device)
# %%
