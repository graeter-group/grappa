#%%
"""
Evaluate a grappa dataset for a model.
"""
from grappa.data import Dataset
from grappa.data import GraphDataLoader
import yaml
import json
from pathlib import Path
from grappa.utils.loading_utils import model_from_tag
from grappa.training.evaluation import Evaluator
import torch
from grappa.models.energy import Energy

#%%
DSNAMES = ['protein-torsion', 'spice-des-monomers']

PURE_TEST_SETS = ['rna-trinucleotide']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download a dataset if not present already:
datasets = [Dataset.from_tag(dsname) for dsname in DSNAMES]


splitpath = str(Path(__file__).parent.parent.parent/f"dataset_creation/get_espaloma_split/espaloma_split.json")

with open(splitpath, "r") as f:
    splitnames = json.load(f)
    # rename the keys to match the Dataset.split() interface:
    splitnames['train_ids'] = splitnames.pop('train')
    splitnames['val_ids'] = splitnames.pop('val')
    splitnames['test_ids'] = splitnames.pop('test')

splitnames.keys()

tr_sets, vl_sets, te_sets = [], [], []
for ds in datasets:
    tr, vl, te = ds.split(**splitnames)
    tr_sets.append(tr)
    vl_sets.append(vl)
    te_sets.append(te)

te_sets += [Dataset.from_tag(dsname) for dsname in PURE_TEST_SETS]

te_set_names = DSNAMES + PURE_TEST_SETS

model = model_from_tag('latest').to(device)
model = torch.nn.Sequential(
    model,
    Energy(),
)

#%%

evaluator = Evaluator()

# For more efficient data loading, we use a GraphDataLoader
for dsname, te_set in zip(te_set_names, te_sets):
    print(f"Evaluating dataset {dsname}")
    if 'rna' in dsname:
        batch_size = 1
    else:
        batch_size = 16
    loader = GraphDataLoader(te_set, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False, conf_strategy='all')

    for g, dsnames in loader:
        g = g.to(device)
        g = model(g)
        evaluator.step(g, dsnames)

print(json.dumps(evaluator.pool(n_bootstrap=10), indent=4))

# %%
