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

#NOTE inint wandb before starting train run. write default oconfig there and provide cmds to overwrite it

#%%
def do_trainrun(config:Dict, project:str='grappa'):
    """
    Do a single training run with the given configuration.
    """

    # check whether all config args are allowed:
    default_config_ = default_config()
    for k in config.keys():
        if k not in default_config_:
            raise KeyError(f"Key {k} not an allowed config argument.")
        if isinstance(config[k], dict):
            for kk in config[k].keys():
                if kk not in default_config_[k]:
                    raise KeyError(f"Key {k}-{kk} not an allowed config argument.")
                

    ###############################
    class ParamFixer(torch.nn.Module):
        def forward(self, g):
            g.nodes['n2'].data['k'] = g.nodes['n2'].data['k'][:,0]
            g.nodes['n2'].data['eq'] = g.nodes['n2'].data['eq'][:,0]
            g.nodes['n3'].data['k'] = g.nodes['n3'].data['k'][:,0]
            g.nodes['n3'].data['eq'] = g.nodes['n3'].data['eq'][:,0]
            return g

    ###############################


    # Get the dataloaders
    tr_loader, val_loader, test_loader = get_dataloaders(**config['data_config'])

    # Get the model
    model = deploy.model_from_config(config=config['model_config'], stat_dict=get_stat_dict(tr_loader))

    # add energy calculation
    model = torch.nn.Sequential(
        model,
        ParamFixer(),
        Energy(suffix=''),
        Energy(suffix='_ref', write_suffix="_classical_ff")
    )

    # Get a pytorch lightning model
    lit_model = LitModel(model=model, tr_loader=tr_loader, vl_loader=val_loader, te_loader=test_loader, **config['lit_model_config'])

    # Initialize wandb
    wandb.init(project=project, config=config)

    # Get the trainer
    trainer = get_lightning_trainer(**config['trainer_config'])

    # write the current config to the run directory. NOTE: initialize a logging dir for the trainer first!
    utils.run_utils.write_yaml(config, Path(trainer.logger.experiment.dir)/"grappa_config.yaml")

    print(f"\nStarting training run {wandb.run.name}...\nSaving logs to {wandb.run.dir}...\n")

    # Train the model
    trainer.fit(model=lit_model, train_dataloaders=tr_loader, val_dataloaders=val_loader)

    return model


def trainrun_from_file(config_path:Path):
    """
    Do a single training run with the configuration in the given file.
    """
    config = load_yaml(config_path)
    do_trainrun(config)