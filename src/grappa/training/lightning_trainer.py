import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path


def get_lightning_trainer(max_epochs=500, gradient_clip_val=1e1, profiler="simple", early_stopping_criterion='early_stopping_loss')->pl.Trainer:

    # Generate a unique ID for the run
    run_id = wandb.util.generate_id()

    # Initialize Wandb with a custom name (using the generated ID)
    wandb.init(name=run_id)

    # Initialize WandbLogger with the existing run
    wandb_logger = WandbLogger(experiment=wandb.run)

    # Checkpoint directory in the run directory
    wandb_run_dir = wandb_logger.experiment.dir
    checkpoint_dir = str(Path(wandb_run_dir) / "checkpoints")

    # ModelCheckpoint callback with dynamic path and filename
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=early_stopping_criterion,
        dirpath=checkpoint_dir,
        filename='best-model',
        save_top_k=1,
        mode='min',
        save_last=True,
        every_n_epochs=10,
    )

    trainer = pl.Trainer(logger=wandb_logger, gradient_clip_val=gradient_clip_val, max_epochs=max_epochs, profiler=profiler, callbacks=[checkpoint_callback])

    return trainer