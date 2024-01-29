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

#%%
PROJECT = "test_opt-scan"

WANDPATH = Path(__file__).parent/'wandb'

RUN_ID = "tbxdr09k"

DEVICE = "cuda"

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

print(f'Evaluating test set with {len(full_test_set)} molecules')

# get the test loader:
test_loader = GraphDataLoader(full_test_set, batch_size=1, conf_strategy='all')
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

