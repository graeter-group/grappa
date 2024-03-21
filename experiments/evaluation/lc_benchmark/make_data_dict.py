#%%
import wandb, torch, numpy as np, matplotlib.pyplot as plt
from grappa.data import Dataset, GraphDataLoader
import torch
from grappa.training.evaluation import Evaluator
from pathlib import Path
import yaml
import json
import wandb
from grappa.training.resume_trainrun import get_dir_from_id
from grappa.models.deploy import model_from_config
from grappa.utils.train_utils import remove_module_prefix
from grappa.models.energy import Energy
from grappa.training.get_dataloaders import get_dataloaders
import copy
import argparse

PROJECT = 'leif-seute/benchmark-grappa-1.1-lc'
DEVICE = 'cuda'

PROJECT_DIR = Path(__file__).parent.parent.parent/'benchmark_experiments'

WANDPATH = PROJECT_DIR/'wandb'

MODELNAME = 'best-model.ckpt'

WITH_TRAIN = False

FORCES_PER_BATCH = 2e3
BATCH_SIZE = None # if None, it will be calculated from FORCES_PER_BATCH

N_BOOTSTRAP = 1


#%%

api = wandb.Api()
runs = api.runs(PROJECT)

data = {}

if Path('results.json').exists():
    with open('results.json', 'r') as f:
        data = json.load(f)

for run in runs:
    RUN_ID = run.id

    print(f'Job {RUN_ID}')

    rundir = get_dir_from_id(run_id=RUN_ID, wandb_folder=WANDPATH)
    modelpath = str(Path(rundir)/'files'/'checkpoints'/MODELNAME)

    configpath = str(Path(rundir)/'files'/'grappa_config.yaml')

    # load a model without weights:
    model = model_from_config(yaml.safe_load(open(configpath))["model_config"]).to(DEVICE).eval()

    # add the energy layer:
    model = torch.nn.Sequential(model, Energy())

    state_dict = torch.load(modelpath, map_location=DEVICE)['state_dict']
    state_dict = remove_module_prefix(state_dict)
    model.load_state_dict(state_dict)

    # load the datasets:
    data_config = yaml.safe_load(open(configpath))["data_config"]

    data_config['save_splits'] = None

    train_loader,_,test_loader = get_dataloaders(**data_config)
    print('loaded test set of size', len(test_loader.dataset))
    print('loaded train set of size', len(train_loader.dataset))

    # evaluator = Evaluator()
    # for i, (g, dsnames) in enumerate(test_loader):
    #     print(f'evaluating {i+1}/{len(test_loader)}', end='\r')
    #     with torch.no_grad():
    #         pred = model(g.to(DEVICE))
    #         evaluator.step(pred, dsnames)

    # print('\n\n')

    # this_data = evaluator.pool(n_bootstrap=N_BOOTSTRAP)
    this_data = data[RUN_ID]

    this_data['test_mols'] = len(test_loader.dataset)
    this_data['train_mols'] = len(train_loader.dataset)

    data[RUN_ID] = this_data
    with open('results.json', 'w') as f:
        json.dump(data, f, indent=4)
# %%
# full run_id:
run_id, num_train_mols = zip(*[(k, data[k]['train_mols']) for k in data.keys()])

# max run id:
max_run_id = run_id[np.argmax(num_train_mols)]

# overwrite the full train molecule state with the final grappa model:
grappa_data = json.load(open('../espaloma_benchmark/results.json', 'r'))
test_mols = data[max_run_id]['test_mols']
train_mols = data[max_run_id]['train_mols']
data[max_run_id] = {ds: grappa_data['test'][ds]['grappa'] for ds in grappa_data['test'].keys() if 'grappa' in grappa_data['test'][ds].keys()}
data[max_run_id]['test_mols'] = test_mols
data[max_run_id]['train_mols'] = train_mols

with open('new_results.json', 'w') as f:
    json.dump(data, f, indent=4)
# %%
