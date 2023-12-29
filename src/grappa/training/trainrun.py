#%%
from grappa.models import Energy, deploy

from grappa.utils.run_utils import write_yaml
import torch
from pathlib import Path
import wandb
from grappa.utils.run_utils import load_yaml
from grappa.training.torchhub_upload import remove_module_prefix
from grappa.utils import graph_utils, dgl_utils
from typing import List, Dict, Union
from grappa.training.get_dataloaders import get_dataloaders
from grappa.training.lightning_model import LitModel
from grappa.training.lightning_trainer import get_lightning_trainer
from grappa.training.config import default_config
from grappa.utils.graph_utils import get_param_statistics, get_default_statistics
from typing import Callable

#%%
def do_trainrun(config:Dict, project:str='grappa', config_from_sweep:Callable=None, manual_sweep_config:Callable=None, sweep_config=None, pretrain_path:Union[Path,str]=None): # NOTE: remove sweep_config
    """
    Do a single training run with the given configuration.

    config_from_sweep: function that takes the wandb.config and returns a dict that can be used to update the grappa config. This can be used for updating several config parameters from one sweep param (e.g. global dropout rates) or for applying transformations to the sweep config (e.g. converting a sweep parameter like the learning rate from linear to log scale). This can be e.g.:
        def config_from_sweep(wandb_config):
            config = {}
            config['lit_model_config'] = {}
            config['lit_model_config']['lr'] = wandb_config.lr

            config['model_config'] = {}
            config['model_config']['param_dropout'] = wandb_config.dropout
            config['model_config']['gnn_conv_dropout'] = wandb_config.dropout

            return config

    manual_sweep_config: function that sets wandb.config parameters specified in the sweep to some manual values defined in the function. This can be used for setting the sweep parameters to some known good starting values. Use eg wandb.config.update({'lr': 0.001}, allow_val_change=True) to set the learning rate to 0.001. In this case, the sweep config must be None.
    pretrain_path: path to a checkpoint that is used to initialize the model weights. This can be a lightning checkpoint or the state dict directly.
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


    # Get the trainer  and initialize wandb
    trainer = get_lightning_trainer(**config['trainer_config'], config=config, project=project)

    experiment_dir = trainer.logger.experiment.dir

    if manual_sweep_config is not None:
        manual_sweep_config()

    # update the config according to the config stored at wandb
    if not config_from_sweep is None:
        # get the hp values that were assigned by wandb sweep:
        wandb_stored_config = wandb.config
        updated_config = config_from_sweep(wandb_stored_config)
        assert len(list(updated_config.keys())) > 0

        # overwrite the config values that correspond to them:
        for k in updated_config.keys():
            if not k in config.keys():
                raise ValueError(f'The sweep config provides a key that is not part of the config: {k}')
            for kk in updated_config[k].keys():
                if not kk in config[k].keys():
                    raise ValueError(f'The sweep config provides a key that is not part of the config: {k}/{kk}')
                config[k][kk] = updated_config[k][kk]


    # write the current config to the run directory.
    write_yaml(config, Path(experiment_dir)/"grappa_config.yaml")

    # Get the dataloaders
    tr_loader, val_loader, test_loader = get_dataloaders(**config['data_config'], classical_needed=config['lit_model_config']['log_classical'], in_feat_names=config['model_config']['in_feat_name'], save_splits=Path(experiment_dir)/'split.json')

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
        # Energy(suffix='_ref', write_suffix="_classical_ff", gradients=gradient_needed)
    )

    if pretrain_path is not None:
        d = torch.load(str(pretrain_path))
        if 'state_dict' in d.keys():
            state_dict = d['state_dict']
            state_dict = remove_module_prefix(state_dict)
        else:
            state_dict = d

        model.load_state_dict(state_dict)

    model.train()

    # test whether the model can be applied to some input of the train set:
    if config['test_model']:
        test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        example, _ = next(iter(val_loader))
        example = example.to(test_device)
        model = model.to(test_device)
        example = model(example)
        example_ = dgl_utils.unbatch(example)[0]
        energies_example = graph_utils.get_energies(example_)
        assert not torch.isnan(energies_example).any(), "Model predicts NaN energies."
        assert not torch.isinf(energies_example).any(), "Model predicts inf energies."
        assert len(energies_example) > 0, "Model predicts no energies."

    # Get a pytorch lightning model
    lit_model = LitModel(model=model, tr_loader=tr_loader, vl_loader=val_loader, te_loader=test_loader, **config['lit_model_config'])

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