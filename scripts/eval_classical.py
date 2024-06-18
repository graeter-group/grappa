#%%
from grappa.data import Dataset
from grappa.training.evaluator import eval_ds
from grappa.utils import unflatten_dict
# %%
ds = Dataset.from_tag("dipeptides-300K-amber99")
ds = ds.slice(0, 10)
# %%
g = ds.graphs[0]
print(list(g.nodes['g'].data.keys()))
# %%
metrics, data = eval_ds(ds, ff_name="amber99sbildn")
print(metrics)
print(ds.mol_ids)
    