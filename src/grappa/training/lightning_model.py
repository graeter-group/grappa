from grappa.data import Dataset, GraphDataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Union

from grappa.data import Dataset, GraphDataLoader
from grappa.models import Energy, get_models
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


class LitModel(pl.LightningModule):
    def __init__(self, model, tr_loader, vl_loader, te_loader, 
                 lrs={0: 1e-4, 3: 1e-5, 200: 1e-6, 400: 1e-7}, 
                 start_qm_epochs=5, add_restarts=[150, 350],
                 warmup_steps=int(2e2), classical_epochs=20,
                 energy_weight=1e-5, gradient_weight=1., tuplewise_weight=0.,
                 log_train_interval=5, log_classical=False, log_params=False):
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
            classical_epochs (int): Number of epochs for classical training before switching to quantum mechanics based training.
            energy_weight (float): Weight of the energy component in the loss function.
            gradient_weight (float): Weight of the gradient component in the loss function.
            tuplewise_weight (float): Weight of the tuplewise component in the loss function.
            log_train_interval (int): Interval (in epochs) at which training metrics are logged.
            log_classical (bool): Whether to log classical values during training.
            log_params (bool): Whether to log parameters during training.
        """

        super().__init__()
        
        self.model = model
        self.tuplewise_energy_loss = TuplewiseEnergyLoss()
        self.loss = MolwiseLoss(gradient_weight=0, energy_weight=0, param_weight=1e-3)

        self.lrs = lrs
        self.lr = self.lrs[0]

        self.start_qm_epochs = start_qm_epochs
        self.classical_epochs = classical_epochs

        self.restarts = sorted(list(set([self.start_qm_epochs, self.classical_epochs] + add_restarts)))
        self.warmup_steps = warmup_steps
        self.warmup_step = None
        self.warmup = True

        self.energy_weight = energy_weight
        self.gradient_weight = gradient_weight
        self.tuplewise_weight = tuplewise_weight

        self.train_dataloader = lambda: tr_loader
        self.val_dataloader = lambda: vl_loader
        self.test_dataloader = lambda: te_loader

        self.log_train_interval = log_train_interval

        self.evaluator = Evaluator(log_parameters=log_params, log_classical_values=log_classical)
        self.train_evaluator = Evaluator(log_parameters=log_params, log_classical_values=log_classical)


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

        if self.current_epoch % self.log_train_interval == 0:
            # log the metrics
            metrics = self.train_evaluator.pool()
            for dsname in metrics.keys():
                for key in metrics[dsname].keys():
                    if any([n in key for n in ["n2", "n3", "n4"]]):
                        self.log(f'parameters/{dsname}/train/{key}', metrics[dsname][key], on_epoch=True)
                    else:
                        self.log(f'{dsname}/train/{key}', metrics[dsname][key], on_epoch=True)
        
        self.set_lr()

        if self.current_epoch > self.classical_epochs:
            self.loss.param_weight = 0

        if self.current_epoch > self.start_qm_epochs:
            print(f'setting the loss weights to\n  param_weight: {self.loss.param_weight}\n  gradient_weight: {self.gradient_weight}\n  energy_weight: {self.energy_weight}')
            self.loss.gradient_weight = float(self.gradient_weight)
            self.loss.energy_weight = float(self.energy_weight)
            

        return super().on_train_epoch_end()



    def training_step(self, batch, batch_idx):

        self.assign_lr()

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
            with torch.no_grad():
                self.train_evaluator.step(g, dsnames)

        return loss


    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            g, dsnames = batch
            g = self(g)

            self.evaluator.step(g, dsnames)

            loss = self.loss(g)
            if self.current_epoch > self.start_qm_epochs:
                self.log('losses/val_loss', loss, batch_size=self.batch_size(g), on_epoch=True)


    def on_validation_epoch_end(self):

        metrics = self.evaluator.pool()
        for dsname in metrics.keys():
            for key in metrics[dsname].keys():
                if any([n in key for n in ["n2", "n3", "n4"]]):
                    self.log(f'parameters/{dsname}/val/{key}', metrics[dsname][key], on_epoch=True)
                else:
                    self.log(f'{dsname}/val/{key}', metrics[dsname][key], on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lrs[0])
        return optimizer
