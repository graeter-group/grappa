#%%
from grappa.data import Dataset, GraphDataLoader
from grappa.training.evaluation import ExplicitEvaluator
import json

#%%
ds = Dataset.from_tag('spice-des-monomers')

#%%
metric_dicts = []
loader = GraphDataLoader(ds, batch_size=128, shuffle=True, conf_strategy='max')

evaluator = ExplicitEvaluator(suffix='_gaff-2.11', suffix_ref='_qm')

for batch, dsnames in loader:
    evaluator.step(batch, dsnames)

d = evaluator.pool()

