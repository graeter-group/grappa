# Experiment class inspired by https://github.com/microsoft/protein-frame-flow

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
from grappa.utils.training_utils import to_df
from grappa.utils.run_utils import flatten_dict, unflatten_dict
from grappa.utils.plotting import make_scatter_plots, compare_scatter_plots
from grappa.models import GrappaModel, Energy
from grappa.utils.model_loading_utils import get_model_dir, get_published_csv_path, get_path_from_tag
import pandas as pd
import torch
import logging
import json
import numpy as np

from pathlib import Path
import wandb
from typing import List, Dict, Union, Tuple
from grappa.utils.graph_utils import get_param_statistics, get_default_statistics
import copy


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
    def __init__(self, config:DictConfig, is_train:bool=False, load_data:bool=True):
        self._cfg = copy.deepcopy(config) # store the config for later use
        self._data_cfg = config.data.data_module
        self._model_cfg = config.model
        self._experiment_cfg = config.experiment
        self._train_cfg = config.experiment.train
        self._energy_cfg = config.data.energy
        self.is_train = is_train

        # if config.data has attribute extra_datasets, add them to data_cfg.datasets:
        if hasattr(config.data, 'extra_datasets'):
            self._data_cfg.datasets = self._data_cfg.datasets + config.data.extra_datasets
        if hasattr(config.data, 'extra_train_datasets'):
            self._data_cfg.pure_train_datasets = self._data_cfg.pure_train_datasets + config.data.extra_train_datasets
        if hasattr(config.data, 'extra_val_datasets'):
            self._data_cfg.pure_val_datasets = self._data_cfg.pure_val_datasets + config.data.extra_val_datasets
        if hasattr(config.data, 'extra_test_datasets'):
            self._data_cfg.pure_test_datasets = self._data_cfg.pure_test_datasets + config.data.extra_test_datasets

        # throw an error if energy terms and ref_terms overlap:
        if set(self._energy_cfg.terms) & set(self._data_cfg.ref_terms):
            raise ValueError(f"Energy terms and reference terms must not overlap. Energy terms are predicted by grappa, reference terms by the reference force field. An overlap means that some contributions are counted twice. Found {set(self._energy_cfg.terms) & set(self._data_cfg.ref_terms)}")

        # create a dictionary from omegaconf config:
        data_cfg = OmegaConf.to_container(self._data_cfg, resolve=True)


        if not str(self._experiment_cfg.checkpointer.dirpath).startswith('/'):
            self._experiment_cfg.checkpointer.dirpath = str(Path(REPO_DIR)/self._experiment_cfg.checkpointer.dirpath)
        self.ckpt_dir = Path(self._experiment_cfg.checkpointer.dirpath)
        
        if load_data:
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
        # (will be stored in the model dict, only important for random initialization)
        param_statistics = get_param_statistics(self.datamodule.train_dataloader()) if hasattr(self, 'datamodule') else get_default_statistics()

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

        # Checkpoint directory.
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Checkpoints saved to {self.ckpt_dir}")
        
        # Model checkpoints
        callbacks.append(ModelCheckpoint(**self._experiment_cfg.checkpointer))
        
        # Save config
        cfg_path = self.ckpt_dir / 'config.yaml'
        with open(cfg_path, 'w') as f:
            OmegaConf.save(config=self._cfg, f=f.name)

        # Log the config to wandb
        self._cfg.experiment = self._experiment_cfg

        if self._experiment_cfg.use_wandb:
            logger = WandbLogger(
                **self._experiment_cfg.wandb,
            )

            # NOTE: the following does not work if multiple gpus are present (does not crash but also does not log the config)
            if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
                flat_cfg = dict(flatten_dict(cfg_dict))
                logger.experiment.config.update(flat_cfg)
        else:
            logger = None

        if self._experiment_cfg.ckpt_path is not None:
            if not str(self._experiment_cfg.ckpt_path).startswith('/'):
                self._experiment_cfg.ckpt_path = Path(REPO_DIR)/self._experiment_cfg.ckpt_path

        if hasattr(self._experiment_cfg, 'ckpt_path') and self._experiment_cfg.ckpt_path is not None:
            use_tag_if_possible(self._experiment_cfg.ckpt_path)

        self.trainer = Trainer(
            **self._experiment_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=self._experiment_cfg.progress_bar,
            enable_model_summary=True,
            inference_mode=False, # important for test call, force calculation needs autograd
            devices=1
        )
        self.trainer.fit(
            model=self.grappa_module,
            datamodule=self.datamodule,
            ckpt_path=self._experiment_cfg.ckpt_path
        )


    def test(self, ckpt_dir:Path=None, ckpt_path:Path=None, n_bootstrap:int=10, test_data_path:Path=None, load_split:bool=False, plot:bool=False, gradient_contributions:List[str]=[], ckpt_data_config=None, store_test_data:bool=True, splitpath:Path=None):
        """
        Evaluate the model on the test sets. Loads the weights from a given checkpoint. If None is given, the best checkpoint is loaded.
        Args:
            ckpt_dir: Path, directory containing the checkpoints from which the best checkpoint is loaded
            ckpt_path: Path, path to the checkpoint to load
            n_bootstrap: int, number of bootstrap samples to calculate the uncertainty of the test metrics
            test_data_path: Path, dir where to store the test data .npz file. if None, the test data is stored in the same directory as the checkpoint
            load_split: bool, whether to load the file defining the split for train/validation/test from the checkpoint directory. If False, it can be assumed that the data module is already set up such that this is the case.
            plot: bool, whether to plot the results
            gradient_contributions: List[str], list of energy terms for which to calculate the gradient contributions
            ckpt_data_config: DictConfig, configuration for the data module that was used to train the model
            store_test_data: bool, whether to store the data calculated for the test set
            splitpath: Path, path to the split file for defining the test set manually

        """
        assert not (ckpt_dir is not None and ckpt_path is not None), "Either ckpt_dir or ckpt_path must be provided, but not both."
        if plot:
            assert store_test_data, "If plot is True, store_test_data must be True as well."

        if len(gradient_contributions) > 0:
            # set the gradient_contributions flag in the energy module to calculate the gradient contributions of the specified terms
            self.grappa_module.model[1].gradient_contributions = True


        if load_split:
            if splitpath is not None:
                self.load_split_from_file(splitpath=splitpath)

            else:
                self.load_split(ckpt_dir=ckpt_dir, ckpt_path=ckpt_path, ckpt_data_config=ckpt_data_config)


        if self.trainer is None:
            self.trainer = Trainer(
                **self._experiment_cfg.trainer,
                logger=False,
                enable_progress_bar=self._experiment_cfg.progress_bar,
                enable_model_summary=False,
                inference_mode=False # important for test call, force calculation needs autograd
            )


        if ckpt_path is None:
            if ckpt_dir is None:
                ckpt_dir = self.ckpt_dir

            # find best checkpoint:
            SEP_CHAR=':'
            ckpts = list([c for c in ckpt_dir.glob('*.ckpt') if SEP_CHAR in c.name])
            losses = [float(ckpt.name.split(SEP_CHAR)[-1].strip('.ckpt')) for ckpt in ckpts]
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
            epoch = ckpt_path.name.split('-')[0] if not ckpt_path.stem in ['last', 'best'] else ckpt_path.stem

        epoch = epoch.replace('.ckpt', '')

        # remove the .ckpt

        self.grappa_module.n_bootstrap = n_bootstrap
        if store_test_data:
            self.grappa_module.test_data_path = Path(ckpt_path).parent / 'test_data' / (epoch+'.npz') if test_data_path is None else Path(test_data_path)
            self.grappa_module.test_data_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.grappa_module.test_data_path = None

        self.grappa_module.test_evaluator.contributions = gradient_contributions

        # make the list explicit (instead of a generator) to get a good progress bar
        self.datamodule.te.graphs = list(self.datamodule.te.graphs)

        self.trainer.test(
            model=self.grappa_module,
            datamodule=self.datamodule,
            ckpt_path=ckpt_path
        )

        if not hasattr(self.grappa_module, 'test_summary'):
            logging.warning("No test summary found. Skipping summary generation.")
            return
        summary = self.grappa_module.test_summary
        if not self.datamodule.num_test_mols is None:
            for dsname in summary.keys():
                summary[dsname]['n_mols'] = self.datamodule.num_test_mols[dsname]

        # Save the summary:
        with(open(self.ckpt_dir / f'summary_{epoch}.json', 'w')) as f:
            json.dump(summary, f, indent=4)

        # transform to dataframe such that we get a table:
        df = to_df(summary, short=True)
        full_df = to_df(summary, short=False)

        table = df.to_string()
        logging.info(f"Test summary:\n{table}")
        
        full_table = full_df.to_string(columns=['n_mols', 'n_confs', 'rmse_energies', 'crmse_gradients', 'mae_energies', 'mae_gradients', 'std_energies', 'std_gradients'])
        with open(self.ckpt_dir / f'summary_{epoch}.txt', 'w') as f:
            f.write(full_table)

        # plot the results:
        if plot:
            datapath = self.grappa_module.test_data_path
            if not datapath.exists():
                logging.warning(f"Test data not found at {datapath}. Skipping plotting.")
            else:
                data = np.load(datapath)
                data = unflatten_dict(data)
                make_scatter_plots(data, plot_dir=datapath.parent/'plots', ylabel='Grappa', xlabel='QM')

        



    def eval_classical(self, classical_force_fields:List[str], ckpt_dir:Path=None, ckpt_path:Path=None, n_bootstrap:int=None, test_data_path:Path=None, load_split:bool=False, plot:bool=False, gradient_contributions:List[str]=[], store_test_data:bool=True, splitpath:Path=None):
        """
        Evaluate the performance of classical force fields (with values stored in the dataset) on the test set.
        Args:
            classical_force_fields: List[str], list of force fields to evaluate
            ckpt_dir: Path, directory containing the checkpoints that define the test set if load_split is True
            ckpt_path: Path, path to the checkpoint that defines the test set if load_split is True
            n_bootstrap: int, number of bootstrap samples to calculate the uncertainty of the test metrics
            test_data_path: Path, path to the test data
            load_split: bool, whether to load the file defining the split for train/validation/test from the checkpoint directory. If False, it can be assumed that the data module is already set up such that this is the case.
            plot: bool, whether to plot the results
            gradient_contributions: List[str], list of energy terms for which to calculate the gradient contributions
        """
        assert not (ckpt_dir is not None and ckpt_path is not None), "Either ckpt_dir or ckpt_path must be provided, but not both."

        if plot:
            assert store_test_data, "If plot is True, store_test_data must be True as well."

        if len(classical_force_fields) == 0:
            logging.info("No classical force fields provided. Skipping their evaluation.")
            return
        logging.info(f"Evaluating classical force fields: {', '.join(classical_force_fields)}...")

        if load_split:
            if splitpath is not None:
                self.load_split_from_file(splitpath=splitpath)
            else:
                self.load_split(ckpt_dir=ckpt_dir, ckpt_path=ckpt_path)


        for ff in classical_force_fields:
            ff = str(ff)
            summary, data = eval_ds(self.datamodule.te, ff, n_bootstrap=n_bootstrap, gradient_contributions=gradient_contributions)

            if len(list(summary.keys())) == 0:
                logging.info(f"No data found for {ff}.")
                continue

            if not self.datamodule.num_test_mols is None:
                for dsname in summary.keys():
                    summary[dsname]['n_mols'] = self.datamodule.num_test_mols[dsname]

            ff_test_data_path = Path(ckpt_path).parent / 'test_data' / ff / 'data.npz' if test_data_path is None else Path(test_data_path).parent/ff/'data.npz'
            
            ff_test_data_path.parent.mkdir(parents=True, exist_ok=True)

            with open(Path(ff_test_data_path).parent / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=4)

            with open(Path(ff_test_data_path).parent / 'summary.txt', 'w') as f:
                f.write(to_df(summary, short=False).to_string(columns=['n_mols', 'n_confs', 'rmse_energies', 'crmse_gradients', 'mae_energies', 'mae_gradients', 'std_energies', 'std_gradients']))

            logging.info(f"Test summary for {ff}:\n{to_df(summary, short=True).to_string()}")

            if store_test_data:
                np.savez(ff_test_data_path, **data)
                logging.info(f"Test data saved to {ff_test_data_path}")

            if plot:
                data = unflatten_dict(data)
                make_scatter_plots(data, plot_dir=ff_test_data_path.parent/'plots', ylabel=ff.capitalize(), xlabel='QM', contributions=gradient_contributions)


    @staticmethod
    def compare_forcefields(ckpt_path:Path, forcefields:List[Tuple[str,str]], gradient_contributions:List[str]=[]):
        """
        Compare two force fields by loading data from npz files that were created during test() or eval_classical().
        """
        def get_ff_data(ff):
            if not (Path(ckpt_path).parent / 'test_data' / ff / 'data.npz').exists() and ff.lower() != 'grappa' and ff != '':
                logging.warning(f"No data found for {ff} at {ckpt_path.parent / 'test_data' / ff / 'data.npz'}")
                return None
            data = np.load(ckpt_path.parent / 'test_data' / ff / 'data.npz') if (ff != '' and ff.lower()!="grappa") else None
            if data is None:
                npz_paths = list((Path(ckpt_path).parent/'test_data').glob('*.npz'))
                assert len(npz_paths) == 1, f"Multiple or no npz files found: {npz_paths}"
                data = np.load(npz_paths[0])
            if data is None:
                logging.warning(f"No data found for {ff} at {ckpt_path.parent / 'test_data' / ff / 'data.npz'}")
                return None
            if len(data.keys()) == 0:
                logging.warning(f"No data found for {ff} at {ckpt_path.parent / 'test_data' / ff / 'data.npz'}")
                return None
            return unflatten_dict(data)

        for ff1, ff2 in forcefields:
            ff1_data = get_ff_data(ff1)
            ff2_data = get_ff_data(ff2)
            if any([d is None for d in [ff1_data, ff2_data]]):
                logging.warning(f"Data not found for {ff1} or {ff2} ... Skipping comparison.")
                continue
            if not 'gradient_contributions' in ff1_data.keys() or not 'gradient_contributions' in ff2_data.keys():
                continue

            if ff1 == '':
                ff1 = 'grappa'
            if ff2 == '':
                ff2 = 'grappa'
            dirname = f"{ff1}-{ff2}" if (ff1!='' and ff2!='') else f'{ff1}{ff2}'
            compare_scatter_plots(ff1_data, ff2_data, plot_dir=ckpt_path.parent / 'compare_plots'/dirname, ylabel=ff1.capitalize(), xlabel=ff2.capitalize(), contributions=gradient_contributions)


    def load_split(self, ckpt_dir:Path=None, ckpt_path:Path=None, ckpt_data_config=None):
        """
        Load the split file from the checkpoint directory and use it to create a test set with unseen molecules.
        Otherwise, re-initialize the dataloader with the config args from before.
        """
        assert ckpt_dir is not None or ckpt_path is not None, "If load_split is True, either ckpt_dir or ckpt_path must be provided."
        load_path = ckpt_dir / 'split.json' if ckpt_dir is not None else ckpt_path.parent / 'split.json'
        if load_path.exists():
            logging.info(f"Loading split from {load_path}")
        else:
            logging.warning(f"Split file not found at {load_path}. Inferring from the data module...")
            
            # init a module just to generate a split file with the data settings from the checkpoint:
            # (we do not use this module since the user might have changed the data settings in the evaluation config)
            ckpt_data_module = OmegaConf.to_container(ckpt_data_config.data_module, resolve=True)
            
            load_path = Path(self.ckpt_dir) / 'split.json'
            load_path.parent.mkdir(parents=True, exist_ok=True)

            re_init_datamodule = GrappaData(**ckpt_data_module, save_splits=load_path)
            re_init_datamodule.setup()

        data_cfg = self._data_cfg
        data_cfg.splitpath = str(load_path) if load_path.exists() else None
        data_cfg.partition = [0.,0.,1.] # all the data that is not in the split file is used for testing (since we assume its unseen)
        self.datamodule = GrappaData(**OmegaConf.to_container(data_cfg, resolve=True))
        self.datamodule.setup()


    def load_split_from_file(self, splitpath:Path):
        """
        Load the split from a file and use it to create a test set.
        """
        splitpath = Path(splitpath)
        if not splitpath.exists():
            raise FileNotFoundError(f"Split file not found at {splitpath}")
        data_cfg = self._data_cfg
        data_cfg.splitpath = str(splitpath)
        data_cfg.partition = [0.,0.,1.] # all the data that is not in the split file is used for testing (since we assume its unseen)
        self.datamodule = GrappaData(**OmegaConf.to_container(data_cfg, resolve=True))
        self.datamodule.setup()


def use_tag_if_possible(ckpt_path:Path):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists() and not str(ckpt_path).endswith('.ckpt'):
        url_tags = pd.read_csv(get_published_csv_path(), dtype=str)['tag'].values
        
        if str(ckpt_path) in url_tags:
            logging.info(f"Model {str(ckpt_path)} not found locally. Downloading...")
            return get_path_from_tag(str(ckpt_path))
             
        if str(ckpt_path.parent.parent.resolve().absolute()) == str(get_model_dir().resolve().absolute()):
            potential_tag = ckpt_path.parent.name
            if potential_tag in url_tags:
                logging.info(f"Model {potential_tag} not found locally. Downloading...")
                return get_path_from_tag(potential_tag)
            
    return ckpt_path