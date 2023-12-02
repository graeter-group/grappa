#%%
from grappa.data import Dataset, GraphDataLoader
from grappa.training.evaluation import ExplicitEvaluator
from grappa.utils.run_utils import get_data_path, load_yaml
#%%
dspath = get_data_path()/'peptides'/'dgl_datasets'/'tripeptides'

# ds = Dataset.load(dspath)
ds = Dataset.load(dspath)
loader = GraphDataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
#%%
evaluator = ExplicitEvaluator(suffix='_reference_ff', suffix_ref='_qm')
for g, dsname in loader:
    evaluator.step(g, dsname)
# %%
d = evaluator.pool()
d
# %%
checkpoint_path = '/hits/fast/mbm/seutelf/grappa/tests/checkpoints/whncielx/model-epoch=189-avg/val/rmse_gradients=9.04.ckpt'
checkpoint_path = '/hits/fast/mbm/seutelf/grappa/tests/experiments/checkpoints/autumn-energy-16/model-epoch=499-avg/val/rmse_gradients=6.40.ckpt'

import torch
import pytorch_lightning as pl
from pathlib import Path

chkpt = torch.load(checkpoint_path)

# %%
state_dict = chkpt['state_dict']
#%%
config = Path('/hits/fast/mbm/seutelf/grappa/tests/wandb/run-20231129_184216-whncielx')

config = Path('/hits/fast/mbm/seutelf/grappa/tests/experiments/wandb/run-20231129_173644-tfbiylf2')


config = load_yaml(config/'files'/'grappa_config.yaml')
# %%
from grappa.models.deploy import model_from_config
from grappa.models.Energy import Energy

def remove_module_prefix(state_dict):
    """ Remove the 'model.' prefix in the beginning of the keys from the state dict keys """
    new_state_dict = {}
    for k,v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class ParamFixer(torch.nn.Module):
    def forward(self, g):
        g.nodes['n2'].data['k'] = g.nodes['n2'].data['k'][:,0]
        g.nodes['n2'].data['eq'] = g.nodes['n2'].data['eq'][:,0]
        g.nodes['n3'].data['k'] = g.nodes['n3'].data['k'][:,0]
        g.nodes['n3'].data['eq'] = g.nodes['n3'].data['eq'][:,0]
        return g

model = model_from_config(config=config['model_config'])
model = torch.nn.Sequential(
    model,
    ParamFixer(),
    Energy(suffix=''),
    Energy(suffix='_ref', write_suffix="_classical_ff")
)
state_dict = remove_module_prefix(state_dict)
model.load_state_dict(state_dict)
# %%
model = model.eval()
model = model.cuda()
evaluator = ExplicitEvaluator(suffix='', suffix_ref='_ref', device='cuda', keep_data=True)
for g, dsname in loader:
    g = g.to('cuda')
    with torch.no_grad():
        g = model(g)
    evaluator.step(g, dsname)
# %%
import copy
unpooled_evaluator = copy.deepcopy(evaluator)
d = evaluator.pool()
d
#%%
energy_ref = evaluator.reference_energies['tripeptides']
energy = evaluator.energies['tripeptides']

import matplotlib.pyplot as plt
import numpy as np
plt.scatter(energy_ref.cpu().numpy(), energy.cpu().numpy())
print(np.sqrt(np.mean(np.square(energy_ref.cpu().numpy()-energy.cpu().numpy()))))
# %%
energy_per_mol = unpooled_evaluator.energies['tripeptides']
energy_per_mol_ref = unpooled_evaluator.reference_energies['tripeptides']
rmses = [np.sqrt(np.mean(np.square(energy_ref.cpu().numpy()-energy.cpu().numpy()))) for energy_ref, energy in zip(energy_per_mol_ref, energy_per_mol)]

plt.hist(rmses, bins=20)
plt.show()
#%%
# does not transfer to gaff charges:
dspath = get_data_path()/'dgl_datasets'/'spice-dipeptide'

# ds = Dataset.load(dspath)
ds = Dataset.load(dspath)
loader = GraphDataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
#%%
for g, dsname in loader:
    with torch.no_grad():
        g = g.to('cuda')
        g = model(g)
    evaluator.step(g, dsname)
#%%
d = evaluator.pool()
d
# %%

# %%
