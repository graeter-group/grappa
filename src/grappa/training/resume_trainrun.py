import torch
torch.set_float32_matmul_precision('medium')
torch.set_default_dtype(torch.float32)

from pathlib import Path
import datetime
import re
import yaml
from grappa.training.get_dataloaders import get_dataloaders
from grappa.training.lightning_model import LitModel
from grappa.training.lightning_trainer import get_lightning_trainer
from grappa.models import deploy, Energy

from grappa.utils.run_utils import write_yaml
from typing import Union, Dict


def get_dir_from_id(wandb_folder:str, run_id:str, max_existing_dirs:int=3)->Path:
    """
    Find the local run directory for a given run id.

    max_existing_dirs: The maximum number of directories that may exist for a given run id. Set this to a finite value to prevent starting runs that are constantly failing.
    """

    # iterate over all files in the wandb folder and find the one that contains the run id, a last.cpkt file and the latest starting time.
    # assume dir name structure: run-YYYYMMDD_HHMMSS-<run_id>

    # Initialize variables to track the latest directory
    latest_time = None
    latest_dir = None

    # Regular expression to match the directory pattern and extract the datetime
    pattern = re.compile(r'run-(\d{8}_\d{6})-' + re.escape(run_id))

    match_counter = 0

    # Iterate over directories in the wandb folder
    for dir in Path(wandb_folder).iterdir():
        if dir.is_dir():
            # Match the pattern
            match = pattern.match(dir.name)
            if match:
                match_counter += 1
                if match_counter > max_existing_dirs:
                    raise RuntimeError(f"More than {max_existing_dirs} directories found for run {run_id}. Aborting...")
                # check whether the directory contains a last.ckpt file
                if not (dir / 'files/checkpoints/last.ckpt').exists():
                    print(f"Directory {dir} does not contain a last.ckpt file. Skipping...")
                    continue
                # Extract the datetime from the directory name
                dir_time = datetime.datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
                if latest_time is None or dir_time > latest_time:
                    latest_time = dir_time
                    latest_dir = dir

    if latest_dir:
        print(f"Latest local directory with last.ckpt for run {run_id}:\n\t{latest_dir}")
    else:
        raise RuntimeError(f"No local directory found for run {run_id}")

    return latest_dir


def resume_trainrun(run_id:str, project:str, wandb_folder:Union[Path, str]=Path.cwd() / 'wandb', new_wandb_run:bool=False, overwrite_config:Dict[str,Dict]={}, add_time_limit:float=23.5):
    """
    Loads the data, model and further setup from the directory of a previous run and either resumes training or starts a new wandb run from where the last run started.
    add_time_limit: Adds this to the time limit of the previous run.
    """

    run_dir = get_dir_from_id(wandb_folder=wandb_folder, run_id=run_id)

    with open(run_dir / 'files/grappa_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # add the time limit:
    config['lit_model_config']['time_limit'] += add_time_limit

    if config['data_config']['splitpath'] is None:
        config['data_config']['splitpath'] = run_dir / 'files/split.json'


    # overwrite config with the given config:
    for k in overwrite_config.keys():
        if not k in config.keys():
            raise ValueError(f'The overwrite_config provides a key that is not part of the config: {k}')
        for kk in overwrite_config[k].keys():
            if not kk in config[k].keys():
                raise ValueError(f'The sweep config provides a key that is not part of the config: {k}/{kk}')
            print(f'Setting {k}/{kk} from {config[k][kk]} to {overwrite_config[k][kk]}')
            config[k][kk] = overwrite_config[k][kk]




    # get the model:
    ###############################
    model = deploy.model_from_config(model_config=config['model_config'])

    gradient_needed = config['lit_model_config']['gradient_weight'] != 0.
    # add energy calculation
    model = torch.nn.Sequential(
        model,
        Energy(suffix='', gradients=gradient_needed),
        # Energy(suffix='_ref', write_suffix="_classical_ff", gradients=gradient_needed)
    )
    model.train()
    ###############################

    # initialize a trainer with the previous wandb run:
    trainer = get_lightning_trainer(**config['trainer_config'], project=project, config=config, resume_id=None if new_wandb_run else run_id)

    experiment_dir = Path(trainer.logger.experiment.dir)

    tr, vl, te = get_dataloaders(**config['data_config'], classical_needed=config['lit_model_config']['log_classical'], in_feat_names=config['model_config']['in_feat_name'], save_splits=experiment_dir / 'files/split.json')

    lit_model = LitModel(model,tr_loader=tr, vl_loader=vl, te_loader=te, **config['lit_model_config'])

    # write the new config file:
    write_yaml(config, Path(experiment_dir)/"grappa_config.yaml")


    trainer.fit(model=lit_model, train_dataloaders=tr, val_dataloaders=vl, ckpt_path=run_dir / 'files/checkpoints/last.ckpt')

    return model