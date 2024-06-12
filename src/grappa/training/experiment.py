# Experiment class inspired by https://github.com/microsoft/protein-frame-flow, Copyright (c) Microsoft Corporation.

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
torch.set_float32_matmul_precision('medium')
torch.set_default_dtype(torch.float32)

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from grappa.data.grappa_data import GrappaData
from grappa.training.evaluator import eval_ds
from grappa.training.lightning_model import GrappaLightningModel
from grappa.utils.run_utils import flatten_dict
from grappa.models import GrappaModel, Energy
import wandb
import torch
import logging
import json
import numpy as np

from pathlib import Path
import wandb
from typing import List, Dict, Union
from grappa.utils.graph_utils import get_param_statistics


REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent


class Experiment:
    """
    Experiment class for training and evaluating a Grappa model.
    
    Application: See experiments/train.py or examples/evaluate.py.
    Initialization: Config files as in configs/train.yaml
    
    Experiment config entries:
    - ckpt_path: str, path to the checkpoint to load for a pretrained model (if None, train from scratch)
    - wandb: dict, configuration for wandb logging
    - progress_bar: bool, whether to show a progress bar
    - checkpointer: dict, lightning checkpointing configuration
    """
    def __init__(self, config:DictConfig, is_train:bool=False):
        self._cfg = config
        self._data_cfg = config.data
        self._model_cfg = config.model
        self._experiment_cfg = config.experiment
        self._train_cfg = config.train
        self._energy_cfg = config.energy
        self.is_train = is_train

        # throw an error if energy terms and ref_terms overlap:
        if set(self._energy_cfg.terms) & set(self._data_cfg.ref_terms):
            raise ValueError(f"Energy terms and reference terms must not overlap. Energy terms are predicted by grappa, reference terms by the reference force field. An overlap means that some contributions are counted twice. Found {set(self._energy_cfg.terms) & set(self._data_cfg.ref_terms)}")

        # create a dictionary from omegaconf config:
        data_cfg = OmegaConf.to_container(self._data_cfg, resolve=True)


        if not str(self._experiment_cfg.checkpointer.dirpath).startswith('/'):
            self._experiment_cfg.checkpointer.dirpath = str(Path(REPO_DIR)/self._experiment_cfg.checkpointer.dirpath)
        self.ckpt_dir = Path(self._experiment_cfg.checkpointer.dirpath)
        
        self.datamodule = GrappaData(**data_cfg, save_splits=self.ckpt_dir/'split.json' if self.is_train else None)
        self.datamodule.setup()

        self._init_model()

        self.trainer = None

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

        # init the model and append an energy module to it, which implements the MM functional differentiably
        model = torch.nn.Sequential(
            GrappaModel(**model_cfg, param_statistics=param_statistics),
            Energy(suffix='', **energy_cfg)
        )

        # wrap a lightning model around it (which handles the training procedure)
        self.grappa_module = GrappaLightningModel(model=model, **train_cfg, param_loss_terms=[t for t in self._energy_cfg.terms if t != 'n4_improper'], start_logging=min(self._experiment_cfg.checkpointer.every_n_epochs, self._train_cfg.start_qm_epochs))


    def train(self):

        assert len(self.datamodule.train_dataloader()) > 0, "No training data found. Please check the data configuration."

        callbacks = []

        logger = WandbLogger(
            **self._experiment_cfg.wandb,
        )

        # Checkpoint directory.
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

        if self._experiment_cfg.ckpt_path is not None:
            if not str(self._experiment_cfg.ckpt_path).startswith('/'):
                self._experiment_cfg.ckpt_path = Path(REPO_DIR)/self._experiment_cfg.ckpt_path

        self.trainer = Trainer(
            **self._experiment_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=self._experiment_cfg.progress_bar,
            enable_model_summary=False,
            inference_mode=False # important for test call, force calculation needs autograd
        )
        self.trainer.fit(
            model=self.grappa_module,
            datamodule=self.datamodule,
            ckpt_path=self._experiment_cfg.ckpt_path
        )


    def test(self, ckpt_dir:Path=None, ckpt_path:Path=None, n_bootstrap:int=10, test_data_path:Path=None, load_split:bool=False):
        """
        Evaluate the model on the test sets. Loads the weights from a given checkpoint. If None is given, the best checkpoint is loaded.
        Args:
            ckpt_dir: Path, directory containing the checkpoints from which the best checkpoint is loaded
            ckpt_path: Path, path to the checkpoint to load
            n_bootstrap: int, number of bootstrap samples to calculate the uncertainty of the test metrics
            test_data_path: Path, dir where to store the test data .npz file
            load_split: bool, whether to load the file defining the split for train/validation/test from the checkpoint directory. If False, it can be assumed that the data module is already set up such that this is the case.
        """
        assert not (ckpt_dir is not None and ckpt_path is not None), "Either ckpt_dir or ckpt_path must be provided, but not both."

        if load_split:
            self.load_split(ckpt_dir=ckpt_dir, ckpt_path=ckpt_path)

        if self.trainer is None:
            self.trainer = Trainer(
                **self._experiment_cfg.trainer,
                logger=False,
                enable_progress_bar=self._experiment_cfg.progress_bar,
                enable_model_summary=True,
                inference_mode=False # important for test call, force calculation needs autograd
            )


        if ckpt_path is None:
            if ckpt_dir is None:
                ckpt_dir = self.ckpt_dir

            # find best checkpoint:
            ckpts = list([c for c in ckpt_dir.glob('*.ckpt') if '=' in c.name])
            losses = [float(ckpt.name.split('=')[-1].strip('.ckpt')) for ckpt in ckpts]
            if len(ckpts) == 0:
                # if no checkpoints with = in the name are found, use the first checkpoint
                all_ckpts = list(ckpt_dir.glob('*.ckpt'))
                if len(all_ckpts) == 0:
                    raise RuntimeError(f"No checkpoints found at {ckpt_dir}")
                elif len(all_ckpts) > 1:
                    raise RuntimeError("More than one checkpoint found, but none of them have loss data.")
                else:
                    ckpt_path = all_ckpts[0]
            else:
                ckpt_path = ckpts[losses.index(min(losses))] if self._experiment_cfg.checkpointer.mode == 'min' else ckpts[losses.index(max(losses))]
            logging.info(f"Evaluating checkpoint: {ckpt_path}")

        if test_data_path is not None:
            epoch = ''
        else:
            epoch = ckpt_path.name.split('-')[0].replace("=", "-") if not ckpt_path.stem in ['last', 'best'] else ckpt_path.stem

        self.grappa_module.n_bootstrap = n_bootstrap
        self.grappa_module.test_data_path = Path(ckpt_path).parent / 'test_data' / (epoch+'.npz') if test_data_path is None else Path(test_data_path)

        self.trainer.test(
            model=self.grappa_module,
            datamodule=self.datamodule,
            ckpt_path=ckpt_path
        )

        summary = self.grappa_module.test_summary
        if not self.datamodule.num_test_mols is None:
            for dsname in summary.keys():
                summary[dsname]['n_mols'] = self.datamodule.num_test_mols[dsname]

        # Save the summary:
        with(open(self.ckpt_dir / f'summary_{epoch}.json', 'w')) as f:
            json.dump(summary, f, indent=4)

        wandb_summary = {}
        for k, v in summary.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    wandb_summary[f"{k}/test/{kk}"] = round(vv['mean'], 3) if isinstance(vv, dict) else round(vv, 3)


        # we do not log via wandb because this will create a chart for each test metric
        logging.info("Test summary:\n\n" + "\n".join([f"{k}:{' '*max(1, 50-len(k))}{v}" for k, v in wandb_summary.items()]))


    def eval_classical(self, classical_force_fields:List[str], ckpt_dir:Path=None, ckpt_path:Path=None, n_bootstrap:int=10, test_data_path:Path=None, load_split:bool=False):
        """
        Evaluate the performance of classical force fields (with values stored in the dataset) on the test set.
        Args:
            classical_force_fields: List[str], list of force fields to evaluate
            ckpt_dir: Path, directory containing the checkpoints that define the test set if load_split is True
            ckpt_path: Path, path to the checkpoint that defines the test set if load_split is True
            n_bootstrap: int, number of bootstrap samples to calculate the uncertainty of the test metrics
            test_data_path: Path, path to the test data
            load_split: bool, whether to load the file defining the split for train/validation/test from the checkpoint directory. If False, it can be assumed that the data module is already set up such that this is the case.
        """
        assert not (ckpt_dir is not None and ckpt_path is not None), "Either ckpt_dir or ckpt_path must be provided, but not both."

        logging.info(f"Evaluating classical force fields: {', '.join(classical_force_fields)}...")

        if load_split:
            self.load_split(ckpt_dir=ckpt_dir, ckpt_path=ckpt_path)

        for ff in classical_force_fields:
            ff = str(ff)
            summary, data = eval_ds(self.datamodule.te, ff, n_bootstrap=None)

            if len(list(summary.keys())) == 0:
                logging.info(f"No data found for {ff}.")
                continue

            summary = self.grappa_module.test_summary
            if not self.datamodule.num_test_mols is None:
                for dsname in summary.keys():
                    summary[dsname]['n_mols'] = self.datamodule.num_test_mols[dsname]

            ff_test_data_path = Path(ckpt_path).parent / 'test_data' / ff / 'data.npz' if test_data_path is None else Path(test_data_path)
            
            ff_test_data_path.parent.mkdir(parents=True, exist_ok=True)

            with open(Path(ff_test_data_path).parent / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=4)

            np.savez(ff_test_data_path, **data)



    def load_split(self, ckpt_dir:Path=None, ckpt_path:Path=None):
        """
        Load the split file from the checkpoint directory and use it to create a test set with unseen molecules.
        """
        assert ckpt_dir is not None or ckpt_path is not None, "If load_split is True, either ckpt_dir or ckpt_path must be provided."
        load_path = ckpt_dir / 'split.json' if ckpt_dir is not None else ckpt_path.parent / 'split.json'
        data_cfg = self._data_cfg
        data_cfg.splitpath = str(load_path)
        data_cfg.partition = [0.,0.,1.] # all the data that is not in the split file is used for testing (since its unseen)
        self.datamodule = GrappaData(**OmegaConf.to_container(data_cfg, resolve=True))
        self.datamodule.setup()