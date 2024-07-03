#%%
from grappa.data import Dataset, GraphDataLoader
from grappa.training.evaluator import Evaluator
import json

#%%
ds = Dataset.from_tag('dipeptides-300K-amber99')

#%%
metric_dicts = []
for batchsize in [1,10,128]:
    loader = GraphDataLoader(ds, batch_size=batchsize, shuffle=True, conf_strategy='max')

    evaluator = Evaluator(suffix='_amber99sbildn_total', suffix_ref='_qm')
    
    for batch, dsnames in loader:
        evaluator.step(batch, dsnames)

    d = evaluator.pool()

    print(json.dumps(d, indent=2))

    metric_dicts.append(d)


# %%
import numpy as np
assert np.allclose(np.array(list(metric_dicts[0]['dipeptides-300K-amber99'].values())), np.array(list(metric_dicts[1]['dipeptides-300K-amber99'].values())))
assert np.allclose(np.array(list(metric_dicts[0]['dipeptides-300K-amber99'].values())), np.array(list(metric_dicts[2]['dipeptides-300K-amber99'].values())))
# %%
