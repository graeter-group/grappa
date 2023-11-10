#%%
from grappa.data import MolData, Dataset
from pathlib import Path
import numpy as np
import torch

dspath = Path(__file__).parents[1]/'data'/"datasets"/"spice-des-monomers"
# %%
mols = []
for i, f in enumerate(dspath.iterdir()):
    if f.suffix == '.npz':
        data = np.load(f)
        data = {k:v for k,v in data.items()}
        moldata = MolData.from_data_dict(data, partial_charge_key='am1bcc_elf_charges', forcefield='openff_unconstrained-1.2.0.offxml')
        mols.append(moldata)
        if i >= 20:
            break
# %%
# duplicate some mols
mols = mols + mols[:10]
ds=Dataset.Dataset.from_moldata(mols, subdataset='monomers')
print(len(ds))
# %%
import tempfile

with tempfile.TemporaryDirectory() as tmpdirname:
    ds.save(tmpdirname)
    ds_loaded = Dataset.Dataset.load(tmpdirname)
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
