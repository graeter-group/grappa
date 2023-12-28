#%%
import wandb

# Authenticate with W&B (ensure you have the API key set up)
wandb.login()

# Define your W&B entity and project
entity = "leif-seute"  # Replace with your W&B entity
project = "test_resumption"  # Replace with your W&B project
wandb_folder = "./wandb"

# Initialize the W&B API
api = wandb.Api()

# Fetch runs from the specified project
runs = api.runs(f"{entity}/{project}")

#%%
# Iterate over runs and check their status
stopped_runs = []
for run in runs:
    if run.state == "failed":
        print(f"Found a failed run: {run.name} (ID: {run.id})")
    stopped_runs.append(run)
    if run.state == 'crashed':
        print(f"Found a crashed run: {run.name} (ID: {run.id})")
    stopped_runs.append(run)
# %%

# find the local run directory:
from pathlib import Path
import datetime
import re
import yaml
from grappa.training.get_dataloaders import get_dataloaders
from grappa.training.lightning_model import LitModel
from grappa.training.lightning_trainer import get_lightning_trainer
from grappa.models import deploy, Energy
import torch
from typing import Union


def get_dir_from_id(wandb_folder:str, run_id:str)->Path:
    """
    Find the local run directory for a given run id.
    """

    # iterate over all files in the wandb folder and find the one that contains the run id, a last.cpkt file and the latest starting time.
    # assume dir name structure: run-YYYYMMDD_HHMMSS-<run_id>

    # Initialize variables to track the latest directory
    latest_time = None
    latest_dir = None

    # Regular expression to match the directory pattern and extract the datetime
    pattern = re.compile(r'run-(\d{8}_\d{6})-' + re.escape(run_id))

    # Iterate over directories in the wandb folder
    for dir in Path(wandb_folder).iterdir():
        if dir.is_dir():
            # Match the pattern
            match = pattern.match(dir.name)
            if match:
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
        print(f"No local directory found for run {run_id}")

    return latest_dir


def resume_trainrun(run_id:str, project:str, wandb_folder:Union[Path, str]=Path.cwd() / 'wandb'):

    run_dir = get_dir_from_id(wandb_folder=wandb_folder, run_id=run_id)

    with open(run_dir / 'files/grappa_config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config['data_config']['splitpath'] is None:
        config['data_config']['splitpath'] = run_dir / 'files/split.json'

    tr, vl, te = get_dataloaders(**config['data_config'], classical_needed=config['lit_model_config']['log_classical'], in_feat_names=config['model_config']['in_feat_name'], save_splits=None)


    # get the model:
    ###############################
    model = deploy.model_from_config(model_config=config['model_config'])

    gradient_needed = config['lit_model_config']['gradient_weight'] != 0.
    # add energy calculation
    model = torch.nn.Sequential(
        model,
        Energy(suffix='', gradients=gradient_needed),
        Energy(suffix='_ref', write_suffix="_classical_ff", gradients=gradient_needed)
    )
    model.train()
    lit_model = LitModel(model,tr_loader=tr, vl_loader=vl, te_loader=te, **config['lit_model_config'])
    ###############################

    trainer = get_lightning_trainer(**config['trainer_config'], resume_id=run_id, project='test_resumption')

    trainer.fit(model=lit_model, train_dataloaders=tr, val_dataloaders=vl, ckpt_path=run_dir / 'files/checkpoints/last.ckpt')


run = stopped_runs[0]
run_id = run.id

run_id = 'n0ufcki1'

resume_trainrun(run_id=run_id, project=project, wandb_folder=wandb_folder)