#%%
from grappa.models import Energy, deploy

import torch
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
import wandb
from grappa.utils.run_utils import get_rundir, load_yaml
from grappa import utils
from typing import List, Dict
from grappa.training.get_dataloaders import get_dataloaders
from grappa.training.lightning_model import LitModel
from grappa.training.lightning_trainer import get_lightning_trainer
from grappa.training.config import default_config
from grappa.utils.graph_utils import get_param_statistics, get_default_statistics

#NOTE inint wandb before starting train run. write default oconfig there and provide cmds to overwrite it

#%%
def do_trainrun(config:Dict, project:str='grappa'):
    """
    Do a single training run with the given configuration.
    """

    # check whether all config args are allowed (they are allowed if they are in the default config)
    default_config_ = default_config()
    for k in config.keys():
        if k not in default_config_:
            raise KeyError(f"Key {k} not an allowed config argument.")
        if isinstance(config[k], dict):
            for kk in config[k].keys():
                if kk not in default_config_[k]:
                    raise KeyError(f"Key {k}-{kk} not an allowed config argument.")

    # Get the dataloaders
    tr_loader, val_loader, test_loader = get_dataloaders(**config['data_config'])

    param_statistics = get_param_statistics(tr_loader)
    for m in ['mean', 'std']:
        for k, v in param_statistics[m].items():
            if torch.isnan(v).any():
                param_statistics[m][k] = get_default_statistics()[m][k]

    # Get the model
    model = deploy.model_from_config(model_config=config['model_config'], param_statistics=param_statistics)

    gradient_needed = config['lit_model_config']['gradient_weight'] != 0.
    # add energy calculation
    model = torch.nn.Sequential(
        model,
        Energy(suffix='', gradients=gradient_needed),
        Energy(suffix='_ref', write_suffix="_classical_ff", gradients=gradient_needed)
    )

    model.train()

    # test whether the model can be applied to some input of the train set:
    if config['test_model']:
        test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        example, _ = next(iter(val_loader))
        example = example.to(test_device)
        model = model.to(test_device)
        example = model(example)
        energies_example = example.nodes['g'].data['energy']
        assert not torch.isnan(energies_example).any(), "Model predicts NaN energies."

    # Get a pytorch lightning model
    lit_model = LitModel(model=model, tr_loader=tr_loader, vl_loader=val_loader, te_loader=test_loader, **config['lit_model_config'])

    # Initialize wandb
    wandb.init(project=project, config=config)

    # Get the trainer
    trainer = get_lightning_trainer(**config['trainer_config'], config=config)

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