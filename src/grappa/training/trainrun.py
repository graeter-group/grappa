#%%
from grappa.data import Dataset, GraphDataLoader
from grappa.models import Energy, get_models
from grappa.training.loss import ParameterLoss, EnergyLoss, GradientLoss

import torch
from pathlib import Path
from grappa.utils.torch_utils import root_mean_squared_error, mean_absolute_error, invariant_rmse, invariant_mae
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from grappa import utils
from typing import List, Dict

#%%
def do_trainrun(config:Dict):
    """Do a single training run with the given configuration.

    Args:
        config (Dict): Dictionary containing the configuration for the training run.
    """
    # Get the model
    model = get_models.get_full_model(config['model_config'])

    # Get the dataset
    # configure sampling weights or total number of mols
    dataset = sum([Dataset.load(p) for p in config['datasets']])

    # Get the split ids
    # load if path in config, otherwise generate. For now, always generate it.
    if False:
        pass
    else:
        split_ids = dataset.calc_split_ids((0.8,0.1,0.1))

    tr, vl, te = dataset.split(*split_ids.values())

    # Get the dataloaders
    train_loader = GraphDataLoader(tr, batch_size=config['tr_batch_size'], shuffle=True, num_workers=config['train_loader_workers'], pin_memory=config['pin_memory'], conf_strategy=config['conf_strategy'])
    val_loader = GraphDataLoader(vl, batch_size=config['val_batch_size'], shuffle=False, num_workers=config['val_loader_workers'], pin_memory=config['pin_memory'], conf_strategy=config['conf_strategy'])
    test_loader = GraphDataLoader(te, batch_size=config['te_batch_size'], shuffle=False, num_workers=config['test_loader_workers'], pin_memory=config['pin_memory'], conf_strategy=config['conf_strategy'])
    
