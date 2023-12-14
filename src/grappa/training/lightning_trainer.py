import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pathlib import Path
import torch
from grappa.models.energy import Energy
from grappa.models.grappa import GrappaModel


def get_lightning_trainer(max_epochs=500, gradient_clip_val=1e1, profiler="simple", early_stopping_criterion='early_stopping_loss', config={})->pl.Trainer:

    # Generate a unique ID for the run
    run_id = wandb.util.generate_id()

    # Initialize Wandb with a custom name (using the generated ID)
    wandb.init(name=run_id)

    # Initialize WandbLogger with the existing run
    wandb_logger = WandbLogger(experiment=wandb.run)

    ###############################
    # CHECKPOINTING MODIFICATIONS:
    # SVAVE MODEL WITHOUT ENERGY MODULE
    # WRITE CONFIG DICT IN THE CHECKPOINT

    # Checkpoint directory in the run directory
    wandb_run_dir = wandb_logger.experiment.dir
    checkpoint_dir = str(Path(wandb_run_dir) / "checkpoints")


    class CustomModelCheckpoint(pl.callbacks.ModelCheckpoint):
        def __init__(self, config, **kwargs):
            super().__init__(**kwargs)
            self.config = config

        def on_epoch_end(self, trainer, pl_module):
            # Save the first part of the sequential model's state_dict
            grappa_model = next(iter(pl_module.sequential))
            assert isinstance(grappa_model, GrappaModel), f"Expected model to be of type torch.nn.Sequential(GrappaModel, Energy) but zeroth part of the sequential is not GrappaModel but {type(grappa_model)}"

            torch.save({
                'state_dict': grappa_model.state_dict(),
                'config': self.config,
            }, self.filepath.format(epoch=trainer.current_epoch, **trainer.logged_metrics))

            super().on_epoch_end(trainer, pl_module)


    # ModelCheckpoint callback with dynamic path and filename
    checkpoint_callback = CustomModelCheckpoint(
        monitor=early_stopping_criterion,
        dirpath=checkpoint_dir,
        filename='best-model',
        save_top_k=1,
        mode='min',
        save_last=True,
        every_n_epochs=10,
        config=config
    )

    trainer = pl.Trainer(logger=wandb_logger, gradient_clip_val=gradient_clip_val, max_epochs=max_epochs, profiler=profiler, callbacks=[checkpoint_callback])

    return trainer