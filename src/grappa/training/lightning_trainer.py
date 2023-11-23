import torch
from pathlib import Path
from grappa.utils.torch_utils import root_mean_squared_error, mean_absolute_error, invariant_rmse, invariant_mae
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from grappa import utils
from typing import List, Dict
from grappa.training.get_dataloaders import get_dataloaders
from grappa.utils.run_utils import get_rundir


def get_lightning_trainer(max_epochs=500, gradient_clip_val=1e1, profiler="simple")->pl.Trainer:

    # keep track of the model with best val loss but only after the first restart:
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     monitor='', # NOTE
    #     dirpath='checkpoints/',
    #     filename='model-{epoch:02d}-{val_loss:.2f}',
    #     save_top_k=1,
    #     mode='min',
    #     save_last=True,
    #     every_n_epochs=20,
    # )


    wandb_logger = WandbLogger()

    trainer = pl.Trainer(logger=wandb_logger, gradient_clip_val=gradient_clip_val, max_epochs=max_epochs, profiler=profiler) #, callbacks=[checkpoint_callback])

    return trainer