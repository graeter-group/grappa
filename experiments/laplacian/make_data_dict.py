"""
Splits the dataset into train, validation and test sets according to the k-fold splits used in the runs. Then, for each fold, only keeps the test set loaders and loads the models that belong to the corresponding fold.
Then, loops over these loaders and models and collects the results using the ExplicitEvaluator.

The policy to choose the model is to use the one with the best validation 'early_stopping_loss' value.
There may exists more than one model trained on the same fold, here we also choose according to the val early_stopping_loss.
Since it can occur that the validation set is contamined, if the early_stopping_loss is higher than some value (by default 20), the last model with the lower train loss is chosen instead.
"""
#%%
from grappa.data import Dataset, GraphDataLoader
import torch
from grappa.training.evaluation import ExplicitEvaluator
from pathlib import Path
import yaml
import json
import wandb
from grappa.training.resume_trainrun import get_dir_from_id
from grappa.models.deploy import model_from_config
import numpy as np
from grappa.utils.train_utils import remove_module_prefix
import pandas as pd
from grappa.models.energy import Energy


PROJECT = "benchmark_testrun_1"

WANDPATH = Path(__file__).parent/'wandb'

RUN_ID = "50122neg"
# RUN_ID = "77jm7wdi"

DEVICE = "cpu"

MODELNAME = 'best-model.ckpt'


#%%

api = wandb.Api()
runs = api.runs(PROJECT)

rundir = get_dir_from_id(run_id=RUN_ID, wandb_folder=WANDPATH)
modelpath = str(Path(rundir)/'files'/'checkpoints'/MODELNAME)

configpath = str(Path(rundir)/'files'/'grappa_config.yaml')
#%%

# load a model without weights:
model = model_from_config(yaml.safe_load(open(configpath))["model_config"]).to(DEVICE)

# add the energy layer:
model = torch.nn.Sequential(model, Energy())

state_dict = torch.load(modelpath, map_location=DEVICE)['state_dict']
state_dict = remove_module_prefix(state_dict)
model.load_state_dict(state_dict)

#%%

# load the datasets:
data_config = yaml.safe_load(open(configpath))["data_config"]
datasets = data_config["datasets"]

pure_test_datasets = data_config["pure_test_datasets"]

pure_testset = Dataset()
for dataset in pure_test_datasets:
    pure_testset += Dataset.from_tag(dataset)

print(f'loaded pure test set with {len(pure_testset)} molecules')

ds = Dataset()

for dataset in datasets:
    ds += Dataset.from_tag(dataset)


print(f'loaded ds with {len(ds)} molecules')

#%%

splitpath = Path(configpath).parent / f"split.json"
splitpath = splitpath if splitpath.exists() else data_config["splitpath"]

splitnames = json.load(open(splitpath))

_, _, test_set = ds.split(splitnames['train'], splitnames['val'], splitnames['test'])

full_test_set = test_set + pure_testset

full_test_set.remove_uncommon_features()

print(f'loaded test set with {len(test_set)} molecules')

# get the test loader:
test_loader = GraphDataLoader(full_test_set, batch_size=16, conf_strategy="all")
evaluator = ExplicitEvaluator()


for i, (g, dsnames) in enumerate(test_loader):
    with torch.no_grad():
        g = g.to(DEVICE)
        g = model(g)
        print(f'batch {i+1}/{len(test_loader)}', end='\r')
        evaluator.step(g, dsnames)


# %%

d = evaluator.pool()
print(json.dumps(d, indent=4))


with open('results_grappa.json', 'w') as f:
    json.dump(d, f, indent=4)
# %%

