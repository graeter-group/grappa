#%%
from grappa.data import Dataset, GraphDataLoader
from grappa.models import Energy, deploy
from grappa.training.loss import ParameterLoss, EnergyLoss, GradientLoss

import torch
from pathlib import Path
from grappa.utils.torch_utils import root_mean_squared_error, mean_absolute_error, invariant_rmse, invariant_mae
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from grappa.utils.run_utils import get_rundir, load_yaml
from grappa import utils
from typing import List, Dict
from grappa.training.get_dataloaders import get_dataloaders
from grappa.training.lightning_model import LitModel
from grappa.training.lightning_trainer import get_lightning_trainer
from grappa.training.config import default_config
from grappa.utils.graph_utils import get_stat_dict
#%%

config = default_config()

config['data_config']['datasets'] = [config['data_config']['datasets'][0]]

# Get the dataloaders
tr_loader, val_loader, test_loader = get_dataloaders(**config['data_config'])

#%%

# Get the model
model = deploy.model_from_config(config=config['model_config'], stat_dict=get_stat_dict(tr_loader))

class ParamFixer(torch.nn.Module):
    def forward(self, g):
        g.nodes['n2'].data['k'] = g.nodes['n2'].data['k'][:,0]
        g.nodes['n2'].data['eq'] = g.nodes['n2'].data['eq'][:,0]
        g.nodes['n3'].data['k'] = g.nodes['n3'].data['k'][:,0]
        g.nodes['n3'].data['eq'] = g.nodes['n3'].data['eq'][:,0]
        return g

# add energy calculation
model = torch.nn.Sequential(
    model,
    ParamFixer(),
    Energy(suffix=''),
    Energy(suffix='_ref', write_suffix="_classical_ff")
)
# %%
g, _ = next(iter(tr_loader))
g = model(g)
# %%

assert g.nodes['g'].data['energy'].requires_grad
assert g.nodes['n1'].data['gradient'].requires_grad
# %%
