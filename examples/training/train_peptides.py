#%%
"""
Training a model on the spice-dipeptides dataset.
Grappa uses the hydra package for config management.

The most straight forward way to train grappa models is to run
    python examples/train.py --config-name your_config
from the root directory of the repository with your own config file configs/your_config.yaml. Alternatively, you can overwrite entries of the default arguments, e.g.
    python examples/train.py data.datasets=[spice-dipeptide] data.pure_train_datasets=[] data.pure_test_datasets=[]

You can also train models in jupyter notebooks, as shown below. (The #%% signal cell boundaries.)
"""

from grappa.training import Experiment
from grappa.utils import get_repo_dir
from pathlib import Path
from hydra import initialize, compose
import os
import torch

config_dir = get_repo_dir() / "configs"

# Compute the relative path from the current directory to the config directory (hydra needs it to be relative)
relative_path = os.path.relpath(config_dir, Path.cwd())

#%%
initialize(config_path=relative_path)
#%%

# Get the default config for training
config = compose(config_name="train")

# set the datasets
config.data.datasets = ['spice-dipeptide']
config.data.pure_train_datasets = []
config.data.pure_val_datasets = []
config.data.pure_test_datasets = []

config.experiment.trainer.max_epochs = 50 if torch.cuda.is_available() else 5
config.experiment.trainer.accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

experiment = Experiment(config)
experiment.train()
#%%
experiment.test()