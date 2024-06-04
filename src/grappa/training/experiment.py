# Experiment class inspired by https://github.com/microsoft/protein-frame-flow, Copyright (c) Microsoft Corporation.

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
torch.set_float32_matmul_precision('medium')
torch.set_default_dtype(torch.float32)

from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from grappa.data.grappa_data import GrappaData
from grappa.training.lightning_model import GrappaLightningModel
from grappa.utils.run_utils import flatten_dict
from grappa.models import GrappaModel, Energy
import wandb
import torch
import logging
import json

from pathlib import Path
import wandb
from typing import List, Dict, Union
from grappa.utils.graph_utils import get_param_statistics

REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Experiment:
    """
    Experiment class for training a Grappa model.
    Experiment config:
    - ckpt_path: str, path to the checkpoint to load for a pretrained model (if None, train from scratch)
    - wandb: dict, configuration for wandb logging
    - progress_bar: bool, whether to show a progress bar
    - checkpointer: dict, lightning checkpointing configuration
    """
    def __init__(self, config:DictConfig):
        self._cfg = config
        self._data_cfg = config.data
        self._model_cfg = config.model
        self._experiment_cfg = config.experiment
        self._train_cfg = config.train
        self._energy_cfg = config.energy

        # throw an error if energy terms and ref_terms overlap:
        if set(self._energy_cfg.terms) & set(self._data_cfg.ref_terms):
            raise ValueError(f"Energy terms and reference terms must not overlap. Energy terms are predicted by grappa, reference terms by the reference force field. An overlap means that some contributions are counted twice. Found {set(self._energy_cfg.terms) & set(self._data_cfg.ref_terms)}")

        # create a dictionary from omegaconf config:
        data_cfg = OmegaConf.to_container(self._data_cfg, resolve=True)

        self.datamodule = GrappaData(**data_cfg)
        self.datamodule.setup()

        self._init_model()


    def _init_model(self):
        """
        Loads the Grappa model from the config file and initializes the GrappaLightningModel.
        For initializing the model:
            - Calculates the statistics of MM parameters in the training set
            - Chains the parameter predictor (GrappaModel) with the Energy module that calculates the energy and gradients of conformations differentiably
        """
        # create a dictionary from omegaconf config:
        model_cfg = OmegaConf.to_container(self._model_cfg, resolve=True)
        energy_cfg = OmegaConf.to_container(self._energy_cfg, resolve=True)
        train_cfg = OmegaConf.to_container(self._train_cfg, resolve=True)

        # calculate the statistics of the MM parameters in the training set for scaling NN outputs
        param_statistics = get_param_statistics(self.datamodule.train_dataloader())

        model = torch.nn.Sequential(
            GrappaModel(**model_cfg, param_statistics=param_statistics),
            Energy(suffix='', **energy_cfg)
        )

        self.grappa_module = GrappaLightningModel(model=model, **train_cfg, param_loss_terms=[t for t in self._energy_cfg.terms if t != 'n4_improper'], start_logging=min(self._experiment_cfg.checkpointer.every_n_epochs, self._train_cfg.start_qm_epochs))


    def train(self):

        assert len(self.datamodule.train_dataloader()) > 0, "No training data found. Please check the data configuration."

        callbacks = []

        logger = WandbLogger(
            **self._experiment_cfg.wandb,
        )

        # Checkpoint directory.
        self.ckpt_dir = Path(REPO_DIR)/self._experiment_cfg.checkpointer.dirpath
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Checkpoints saved to {self.ckpt_dir}")
        
        # Model checkpoints
        callbacks.append(ModelCheckpoint(**self._experiment_cfg.checkpointer, filename='{epoch}-{'+self._experiment_cfg.checkpointer.monitor+':.3e}'))
        
        # Save config
        cfg_path = self.ckpt_dir / 'config.yaml'
        with open(cfg_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f.name)
        cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
        flat_cfg = dict(flatten_dict(cfg_dict))
        assert isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config), f"Expected wandb config, but got {type(logger.experiment.config)}"
        logger.experiment.config.update(flat_cfg)

        self.trainer = Trainer(
            **self._experiment_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=self._experiment_cfg.progress_bar,
            enable_model_summary=True,
            inference_mode=False # important for test call, force calculation needs autograd
        )
        self.trainer.fit(
            model=self.grappa_module,
            datamodule=self.datamodule,
            ckpt_path=self._experiment_cfg.ckpt_path
        )


    def test(self):

        # find best checkpoint:
        ckpts = list([c for c in self.ckpt_dir.glob('*.ckpt') if '=' in c.name])
        losses = [float(ckpt.name.split('=')[-1].strip('.ckpt')) for ckpt in ckpts]
        if len(ckpts) == 0:
            # last
            try:
                best_ckpt = list(self.ckpt_dir.glob('*.ckpt'))[0]
            except IndexError:
                logging.error("No checkpoints found.")
                return
        else:
            best_ckpt = ckpts[losses.index(min(losses))] if self._experiment_cfg.checkpointer.mode == 'min' else ckpts[losses.index(max(losses))]
        logging.info(f"Best checkpoint: {best_ckpt}")

        self.trainer.test(
            model=self.grappa_module,
            datamodule=self.datamodule,
            ckpt_path=best_ckpt
        )

        summary = self.grappa_module.test_summary

        # Save the summary:
        with(open(self.ckpt_dir / 'summary.json', 'w')) as f:
            json.dump(summary, f, indent=4)

        wandb_summary = {}
        for k, v in summary.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        wandb_summary[f"{k}/test/{kk}"] = round(vv['mean'], 3)

        # we do not log via wandb because this will create a chart for each test metric
        logging.info("Test summary:\n\n" + "\n".join([f"{k}:{' '*max(1, 50-len(k))}{v}" for k, v in wandb_summary.items()]))