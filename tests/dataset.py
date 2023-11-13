#%%
from grappa.data import MolData, Dataset
from pathlib import Path
import numpy as np
import torch

dspath = Path(__file__).parents[1]/'data'/"grappa_datasets"/"spice-des-monomers"
# %%
mols = []
for i, f in enumerate(dspath.iterdir()):
    if f.suffix == '.npz':
        data = np.load(f)
        data = {k:v for k,v in data.items()}
        moldata = MolData.load(f)
        mols.append(moldata)
        if i >= 20:
            break
# %%
# duplicate some mols
mols = mols + mols[:10]
ds=Dataset.from_moldata(mols, subdataset='monomers')
print(len(ds))
# %%
import tempfile

with tempfile.TemporaryDirectory() as tmpdirname:
    ds.save(tmpdirname)
    ds_loaded = Dataset.load(tmpdirname)
assert len(ds_loaded) == len(ds)
assert torch.all(ds_loaded[0][0].nodes['n1'].data['xyz'] == ds[0][0].nodes['n1'].data['xyz'])
# %%
split = ds.calc_split_ids(partition={'monomers':(0.8, 0.1, 0.1)}, seed=20)
print({k:len(v) for k,v in split.items()})
# %%
tr, vl, te = ds.split(train_ids=split['train'], val_ids=split['val'], test_ids=split['test'], check_overlap=True)
# %%
print(len(tr), len(vl), len(te))
# %%
full = te + vl + tr

# check that the content is the same up to order:
assert len(full) == len(ds)
ids1 = [mol_id for mol_id in full.mol_ids]
graphs1 = [mol[0] for mol in full]
ids2 = [mol_id for mol_id in ds.mol_ids]
graphs2 = [mol[0] for mol in ds]

assert set(ids1) == set(ids2)
for id1 in ids1:
    assert torch.all(graphs1[ids1.index(id1)].nodes['n1'].data['xyz'] == graphs2[ids2.index(id1)].nodes['n1'].data['xyz'])
# %%
# ds[10][0].nodes['n1'].data['sp_hybridization']
# # %%
# ds[10][0].nodes['n1'].data['atomic_number'][:,:10]
# # %%
# ds[10][0].nodes['n1'].data['is_aromatic']
# # %%
# ds[10][0].nodes['n1'].data['ring_encoding']
# # %%
# ds[10][0].nodes['n1'].data['partial_charge']
# %%
from grappa.data import GraphDataLoader
from grappa.utils import dgl_utils

loader = GraphDataLoader(ds, batch_size=10, num_workers=1, shuffle=False)
# %%
for g, dsname in loader.to('cuda'):
    print(f'n_instances: {g.num_nodes("g")}, num atoms: {g.nodes["n1"].data["xyz"].shape[0]}, dsname: {dsname})')
    graphs = dgl_utils.unbatch(g)
    assert len(graphs) == len(dsname)
    assert 'cuda' in str(g.nodes['n1'].data['xyz'].device)

# %%
