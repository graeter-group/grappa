#%%
from grappa.data import Dataset, GraphDataLoader
from grappa.training.evaluation import ExplicitEvaluator
import json

#%%
ds = Dataset.from_tag('spice-des-monomers')

#%%
metric_dicts = []
for batchsize in [1,128]:
    loader = GraphDataLoader(ds, batch_size=batchsize, shuffle=True, conf_strategy='all')

    evaluator = ExplicitEvaluator(suffix='_gaff-2.11', suffix_ref='_qm')
    
    for batch, dsnames in loader:
        evaluator.step(batch, dsnames)

    d = evaluator.pool()

    print(json.dumps(d, indent=2))

    metric_dicts.append(d)


# %%
