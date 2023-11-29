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


def get_lightning_trainer(max_epochs=500, gradient_clip_val=1e1, profiler="simple", early_stopping_criterion='avg/val/rmse_gradients')->pl.Trainer:

    wandb_logger = WandbLogger()

    run_id = wandb_logger.experiment.id

    # Checkpoint directory path using the run name
    checkpoint_dir = f'checkpoints/{run_id}/'

    # ModelCheckpoint callback with dynamic path and filename
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=early_stopping_criterion,
        dirpath=checkpoint_dir,
        filename='model-{epoch:02d}-{'+early_stopping_criterion+':.2f}',
        save_top_k=2,
        mode='min',
        save_last=True,
        every_n_epochs=10,
    )

    trainer = pl.Trainer(logger=wandb_logger, gradient_clip_val=gradient_clip_val, max_epochs=max_epochs, profiler=profiler, callbacks=[checkpoint_callback])

    return trainer