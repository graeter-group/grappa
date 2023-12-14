from grappa.training.loss import MolwiseLoss, TuplewiseEnergyLoss
from grappa.training.evaluation import Evaluator
import numpy as np
import json

import torch
from pathlib import Path
from grappa.utils.torch_utils import root_mean_squared_error, mean_absolute_error, invariant_rmse, invariant_mae
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from grappa import utils
from typing import List, Dict
from grappa.training.get_dataloaders import get_dataloaders
import time


class LitModel(pl.LightningModule):
    def __init__(self, model, tr_loader, vl_loader, te_loader, 
                 lrs={0: 1e-4, 3: 1e-5, 200: 1e-6, 400: 1e-7}, 
                 start_qm_epochs=1, add_restarts=[],
                 warmup_steps=int(2e2),
                 energy_weight=1., gradient_weight=1e-1, tuplewise_weight=0.,
                 param_weight=1e-4, proper_regularisation=1e-5, improper_regularisation=1e-5,
                 log_train_interval=5, log_classical=False, log_params=False, weight_decay=0.,
                 early_stopping_energy_weight=2.,
                 log_metrics=True,
                 patience:int=30, lr_decay:float=0.8, time_limit:float=None):
        """
        Initialize the LitModel with specific configurations.

        Args:
            model (torch.nn.Module): The neural network model to be trained.
            tr_loader (DataLoader): DataLoader for the training dataset.
            vl_loader (DataLoader): DataLoader for the validation dataset.
            te_loader (DataLoader): DataLoader for the test dataset.
            lrs (dict): A dictionary mapping from epoch number to learning rate.
                        Specifies the learning rate to use after each given epoch.
            start_qm_epochs (int): The epoch number from which quantum mechanics based training starts.
            add_restarts (list): List of epochs at which to restart the training process.
            warmup_steps (int): The number of steps over which the learning rate is linearly increased.
            energy_weight (float): Weight of the energy component in the loss function.
            gradient_weight (float): Weight of the gradient component in the loss function.
            tuplewise_weight (float): Weight of the tuplewise component in the loss function.
            param_weight (float): Weight of the parameter component in the loss function. Prefactor of per-mol average of each parameter in the loss. By default, bond and angle parameters are rewighted such that their std deviation is around O(1), thus this is comparable to mse of energy and gradient (averaged molecule-wise).
            proper_regularisation (float): Weight of the proper L2 regularisation component in the loss function.
            improper_regularisation (float): Weight of the improper L2 regularisation component in the loss function.
            log_train_interval (int): Interval (in epochs) at which training metrics are logged.
            log_classical (bool): Whether to log classical values during training.
            log_params (bool): Whether to log parameters during training.
            weight_decay (float): Weight decay parameter for the optimizer.
            early_stopping_energy_weight (float): Weight of the energy component in the early stopping criterion, which is given by the sum of the average gradient rmse per dataset (first average over mols and confs in the dataset for obtaining rmses and then over the datasets for obtaining a mean rmse) and the energy rmse per dataset (same average procedure) multiplied by this weight. (This way, each dataset contributes equally to the early stopping criterion, independent of the number of molecules in the dataset.)
            patience (int): Number of epochs without increase of the early stopping criterion (i.e. the weighted sum of energy/force rmse averaged over the dataset types) after which the learning rate is decreased: The best early stopping criterion value is stored and if the current value is larger than the best value, the number of epochs without improvement is increased by 1, else set to zero. If the number of epochs without improvement is larger than the patience, the learning rate is decreased by a factor of decay.
            lr_decay (float): Factor by which to decrease the learning rate if the early stopping criterion does not improve for patience epochs.
            time_limit (float): Time limit in hours. If the training takes longer than this, the training is stopped.
        """

        if tuplewise_weight > 0:
            raise NotImplementedError("Tuplewise loss is not yet implemented.")

        super().__init__()
        
        self.model = model
        self.tuplewise_energy_loss = TuplewiseEnergyLoss()
        
        # first, set energy and gradient weight to zero to only train the parameters. these are re-setted in on_train_epoch_start
        self.loss = MolwiseLoss(gradient_weight=0, energy_weight=0, param_weight=1e-3 if not param_weight==0 else 0, proper_regularisation=proper_regularisation, improper_regularisation=improper_regularisation)

        self.lrs = lrs
        self.lr = self.lrs[0]
        self.weight_decay = weight_decay

        self.start_qm_epochs = start_qm_epochs

        self.restarts = sorted(list(set([self.start_qm_epochs] + add_restarts)))
        self.warmup_steps = warmup_steps
        self.warmup_step = None
        self.warmup = True

        self.patience = patience
        self.lr_decay = lr_decay

        self.energy_weight = energy_weight
        self.gradient_weight = gradient_weight
        self.tuplewise_weight = tuplewise_weight
        self.param_weight = param_weight

        self.train_dataloader = lambda: tr_loader
        self.val_dataloader = lambda: vl_loader
        self.test_dataloader = lambda: te_loader

        self.log_train_interval = log_train_interval

        self.evaluator = Evaluator(log_parameters=log_params, log_classical_values=log_classical)
        self.train_evaluator = Evaluator(log_parameters=log_params, log_classical_values=log_classical)

        self.early_stopping_energy_weight = early_stopping_energy_weight

        self.log_metrics = log_metrics

        self.time_limit = time_limit


        # helper variables:
        self.best_early_stopping_loss = float("inf")
        self.epochs_without_improvement = 0

        self.time_start = time.time()



    def forward(self, x):
        return self.model(x)

    def batch_size(self, g):
        return g.num_nodes('g')


    def get_lr(self):
        if not self.warmup_step is None:

            if self.warmup_step >= self.warmup_steps:
                self.warmup_step = None
                return self.lr
            
            # increase the lr linearly to self.lr in self.warmup_steps steps
            lr = float(self.warmup_step) / self.warmup_steps * self.lr
            self.warmup_step += 1
            return lr
        
        else:
            return self.lr


    def set_lr(self):
        """
        Sets the learning rate of the optimizer to self.lr. Resets the learning rate (self.lr) and the optimizer after restarts. To be called once per train epoch.
        """

        if self.current_epoch in self.lrs.keys():
            self.lr = self.lrs[self.current_epoch]        
        

        if self.current_epoch in self.restarts:
            self.trainer.optimizers[0] = torch.optim.Adam(self.parameters(), lr=self.lr)
            self.warmup_step = 0


    def assign_lr(self):
        lr = self.get_lr()

        # reset the lr of the current optimizer:
        for param_group in self.trainer.optimizers[0].param_groups:
            param_group['lr'] = lr


    def on_train_epoch_end(self) -> None:
        # log the metrics every log_train_interval epochs:
        if self.log_metrics:
            if self.current_epoch % self.log_train_interval == 0:
                metrics = self.train_evaluator.pool()
                for dsname in metrics.keys():
                    for key in metrics[dsname].keys():
                        if any([n in key for n in ["n2", "n3", "n4"]]):
                            self.log(f'parameters/{dsname}/train/{key}', metrics[dsname][key], on_epoch=True)
                        else:
                            self.log(f'{dsname}/train/{key}', metrics[dsname][key], on_epoch=True)
            
                # Early stopping criterion:
                gradient_avg = metrics["avg"]["rmse_gradients"]
                energy_avg = metrics["avg"]["rmse_energies"]
                early_stopping_loss = self.early_stopping_energy_weight * energy_avg + gradient_avg
                self.log('train_early_stopping_loss', early_stopping_loss, on_epoch=True)

        return super().on_train_epoch_end()
        

    def on_train_epoch_start(self) -> None:
        # Update epoch-dependent hyperparameters such as lr and loss weights.

        self.set_lr()

        if self.current_epoch > self.start_qm_epochs:
            # reset the loss weights to the values specified in the config:
            self.loss.gradient_weight = float(self.gradient_weight)
            self.loss.energy_weight = float(self.energy_weight)
            self.loss.param_weight = float(self.param_weight)
            
        return super().on_train_epoch_start()



    def training_step(self, batch, batch_idx):

        self.assign_lr()

        if self.log_metrics:
            self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

        g, dsnames = batch

        try:
            g = self(g)
        except Exception as e:
            shape_dict = {ntype: {feat: str(g.nodes[ntype].data[feat].shape) for feat in g.nodes[ntype].data.keys()} for ntype in g.ntypes}
            raise ValueError(f"Error in forward pass for batch {batch_idx}, dsnames {dsnames} and graph with feature shapes\n{json.dumps(shape_dict, indent=4)}:\n{e}") from e

        loss = self.loss(g)

        if self.current_epoch > self.start_qm_epochs:
            self.log('losses/train_loss', loss, batch_size=self.batch_size(g), on_epoch=True)

        if self.current_epoch%self.log_train_interval == 0:
            if self.log_metrics:
                with torch.no_grad():
                    self.train_evaluator.step(g, dsnames)

        return loss


    def validation_step(self, batch, batch_idx):
        if self.log_metrics:
            with torch.no_grad():
                g, dsnames = batch
                g = self(g)

                self.evaluator.step(g, dsnames)

                loss = self.loss(g)
                if self.current_epoch > self.start_qm_epochs:
                        self.log('losses/val_loss', loss, batch_size=self.batch_size(g), on_epoch=True)


    def on_validation_epoch_end(self):
        if self.log_metrics:
            metrics = self.evaluator.pool()
            for dsname in metrics.keys():
                for key in metrics[dsname].keys():
                    if any([n in key for n in ["n2", "n3", "n4"]]):
                        self.log(f'parameters/{dsname}/val/{key}', metrics[dsname][key], on_epoch=True)
                    else:
                        self.log(f'{dsname}/val/{key}', metrics[dsname][key], on_epoch=True)

            # Early stopping criterion:
            gradient_avg = metrics["avg"]["rmse_gradients"]
            energy_avg = metrics["avg"]["rmse_energies"]
            early_stopping_loss = self.early_stopping_energy_weight * energy_avg + gradient_avg
            self.log('early_stopping_loss', early_stopping_loss, on_epoch=True)

        if patience > 0:
            assert self.log_metrics, "Early stopping criterion is only implemented if metrics are logged."

            if early_stopping_loss < self.best_early_stopping_loss:
                self.best_early_stopping_loss = float(early_stopping_loss)
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement > self.patience:
                self.lr *= self.lr_decay
                self.epochs_without_improvement = 0

        # stop training if time limit is exceeded:
        if not self.time_limit is None:
            if time.time() - self.time_start > self.time_limit*3600.:
                self.trainer.should_stop = True


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lrs[0], weight_decay=self.weight_decay)
        return optimizer
