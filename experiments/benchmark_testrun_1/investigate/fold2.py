#%%
from pathlib import Path
import wandb
from grappa.training.resume_trainrun import get_dir_from_id
from grappa.training.get_dataloaders import get_dataloaders
import yaml

PROJECT = 'benchmark_testrun_1'

wandbpath = Path(__file__).parent.parent/'wandb'

api = wandb.Api()

runs = api.runs(PROJECT)
# %%
name = 'fold_2'
rundirs = []
for run in runs:
    if name in run.name:
        print(run.name)
        rundir = get_dir_from_id(wandbpath, run.id)
        rundirs.append(rundir)
        print()        
# %%
rundir = rundirs[0]

config = yaml.safe_load((rundir/'files'/'grappa_config.yaml').read_text())

config['data_config']['datasets'] = ['gen2']

config['data_config']['pure_train_datasets'] = []
config['data_config']['pure_val_datasets'] = []
config['data_config']['pure_test_datasets'] = []

tr, vl, te = get_dataloaders(**config['data_config'])
# %%
from grappa.training.torchhub_upload import get_grappa_model

checkpoint_path = rundir/'files'/'checkpoints'/'last.ckpt'

model, _, _ = get_grappa_model(checkpoint_path)
# %%
from grappa.models import Energy
import torch
from grappa.training.evaluation import ExplicitEvaluator

device = 'cuda'
# device = 'cpu'

model = torch.nn.Sequential(model, Energy(suffix=''))
model = model.eval()
model = model.to(device)
evaluator = ExplicitEvaluator(log_classical_values=True, suffix_classical='_gaff-2.11')#keep_data=True)

loader = tr

for i, (g, dsnames) in enumerate(loader):
    print(f'{i+1}/{len(loader)}', end='\r')
    g = g.to(device)
    with torch.no_grad():
        g = model(g)
    g = g.cpu()
    evaluator.step(g, dsnames)
# %%
import json

d = evaluator.pool()
print(json.dumps(d, indent=4))
# %%
import numpy as np
import matplotlib.pyplot as plt

all_grads = np.concatenate([e.numpy().flatten() for e in evaluator.reference_gradients['gen2']])
grads = [e.max() for e in evaluator.reference_gradients['gen2']]
plt.hist(all_grads, bins=100, log=True)
plt.hist(grads, bins=100, log=True)
# %%

#%%
from grappa.utils.dgl_utils import unbatch

crmses = []
for g_, _ in loader:
    graphs = unbatch(g_)
    for g in graphs:
        grad_gaff = g.nodes['n1'].data['gradient_gaff-2.11']
        grad_qm = g.nodes['n1'].data['gradient_qm']
        grad_crmse = torch.sqrt(((grad_gaff - grad_qm)**2).mean())
        crmses.append(grad_crmse.item())

plt.hist(crmses, bins=100)
# %%
