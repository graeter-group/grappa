from grappa.training.loss import MolwiseLoss
from grappa.training.evaluation import FastEvaluator, Evaluator
from grappa.models.energy import Energy
import torch
import pytorch_lightning as pl
from typing import List, Dict
import time
import sys
import logging
import wandb
import json

class GrappaLightningModel(pl.LightningModule):
    def __init__(self,
                 model,
                 lr:float=1e-5, 
                 energy_weight:float=1.,
                 gradient_weight:float=1e-1,
                 start_qm_epochs:int=0,
                 param_loss_epochs:int=100,
                 warmup_steps=500,
                 early_stopping_energy_weight=3.,
                 param_weight=1e-4,
                 proper_regularisation=1e-5,
                 improper_regularisation=1e-5,
                 weight_decay=0.,
                 patience:int=50,
                 lr_decay:float=0.8,
                 tuplewise_weight:float=0,
                 time_limit:float=None,
                 finish_criterion:Dict[int, float]={},
                 param_weights_by_dataset:Dict[str,float]={},
                 param_loss_terms:List[str]=['n2', 'n3', 'n4'],
                 log_train_interval:int=10,
                 start_logging:int=0,
                ):
        """
        Initialize the GrappaLightningModel with specific configurations.

        Args:
            model (torch.nn.Module): The neural network model to be trained.
            lr (float): Learning rate for the optimizer. Defaults to 1e-5.
            energy_weight (float): Weight of the energy component in the loss function. Defaults to 1.
            gradient_weight (float): Weight of the gradient component in the loss function. Defaults to 1e-1.
            start_qm_epochs (int): The epoch number from which quantum mechanics based training starts. Defaults to 1.
            param_loss_epochs (int): Epoch number from which the parameter and tuplewise loss weights are set to zero and the optimizer is restarted. Defaults to 100.
            warmup_steps (int): The number of steps over which the learning rate is linearly increased. Defaults to 200.
            early_stopping_energy_weight (float): Weight of the energy component in the early stopping criterion. Defaults to 3.
            param_weight (float): Weight of the parameter component in the loss function. Defaults to 1e-4.
            proper_regularisation (float): Weight of the proper L2 regularisation component in the loss function. Defaults to 1e-5.
            improper_regularisation (float): Weight of the improper L2 regularisation component in the loss function. Defaults to 1e-5.
            weight_decay (float): Weight decay parameter for the optimizer. Defaults to 0.
            patience (int): Number of epochs without improvement of the early stopping criterion after which the learning rate is decreased. Defaults to 30.
            lr_decay (float): Factor by which to decrease the learning rate if the early stopping criterion does not improve for patience epochs. Defaults to 0.8.
            tuplewise_weight (float): Weight of the tuplewise component in the loss function. Defaults to 0.
            time_limit (float, optional): Time limit in hours. If the training takes longer than this, the training is stopped. Defaults to None.
            finish_criterion (Dict[int, float], optional): Dictionary mapping from time in hours to maximum early stopping criterion value. Defaults to {}.
            param_weights_by_dataset (Dict[str, float], optional): Dictionary mapping from dataset name to weight of the parameter loss for this dataset. Defaults to {}.
            param_loss_terms (List[str], optional): Which parameters to train on directly during epoch < param_loss_epochs. Defaults to ['n2', 'n3', 'n4'].
            log_train_interval (int, optional): Interval in epochs for logging the training metrics (instead of logging every epoch since this slows down the train steps). Defaults to 10.
        """
        super().__init__()
        
        self.model = model
        
        # first, set energy and gradient weight to zero to only train the parameters. these are re-setted in on_train_epoch_start
        if start_qm_epochs > 0:
            self.loss_fn = MolwiseLoss(gradient_weight=0, energy_weight=0, param_weight=1e-3, tuplewise_weight=tuplewise_weight, proper_regularisation=proper_regularisation, improper_regularisation=improper_regularisation, param_weights_by_dataset=param_weights_by_dataset, terms=param_loss_terms)
        
        else:
            self.loss_fn = MolwiseLoss(gradient_weight=gradient_weight, energy_weight=energy_weight, param_weight=param_weight, tuplewise_weight=tuplewise_weight, proper_regularisation=proper_regularisation, improper_regularisation=improper_regularisation, param_weights_by_dataset=param_weights_by_dataset, terms=param_loss_terms)

        self.lr = lr
        self.weight_decay = weight_decay

        self.start_qm_epochs = start_qm_epochs

        self.warmup_steps = warmup_steps
        self.warmup_step = None
        self.warmup = True

        self.patience = patience
        self.lr_decay = lr_decay

        self.energy_weight = energy_weight
        self.gradient_weight = gradient_weight
        self.tuplewise_weight = tuplewise_weight
        self.param_weight = param_weight

        self.param_loss_epochs = param_loss_epochs
        self.restarts = [self.param_loss_epochs] if not self.param_loss_epochs is None else []

        self.finish_criterion = finish_criterion

        self.evaluator = FastEvaluator()
        self.train_evaluator = FastEvaluator()
        self.test_evaluator = Evaluator()

        self.early_stopping_energy_weight = early_stopping_energy_weight

        self.time_limit = time_limit

        self.log_train_interval = log_train_interval
        self.start_logging = start_logging

        self.elapsed_time = 0 # time from a previous run, if the training is restarted.  it is checked whether time.time() - start_time + elaped_time > time_limit in on_validation_epoch_end

        # helper variables:
        self.best_early_stopping_loss = float("inf")
        self.epochs_without_improvement = 0

        self.time_start = time.time()


    def forward(self, x):
        return self.model(x)

    def batch_size(self, g):
        return g.num_nodes('g')

    def _get_lr(self):
        """
        Manages warmup, slowly increases from zero to self.lr linearly.
        """
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


    def assign_lr(self):
        lr = self._get_lr()

        # reset the lr of the current optimizer:
        for param_group in self.trainer.optimizers[0].param_groups:
            param_group['lr'] = lr


    def on_train_epoch_end(self) -> None:
        # log the metrics every log_train_interval epochs:
        if self.current_epoch > self.start_qm_epochs and self.current_epoch % self.log_train_interval == 0:
            metrics = self.train_evaluator.pool()
            for dsname in metrics.keys():
                for key in metrics[dsname].keys():
                    self.log(f'{dsname}/train/{key}', metrics[dsname][key], on_epoch=True)

            # Early stopping criterion:
            gradient_avg = metrics["avg"]["rmse_gradients"]
            energy_avg = metrics["avg"]["rmse_energies"]
            early_stopping_loss = self.early_stopping_energy_weight * energy_avg + gradient_avg
            self.log('train_early_stopping_loss', early_stopping_loss, on_epoch=True)
    
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

        return super().on_train_epoch_end()
        

    def on_train_epoch_start(self) -> None:

        self._epoch_start_time = time.time()

        # Update epoch-dependent hyperparameters such as lr and loss weights.

        if self.current_epoch in self.restarts:
            self.trainer.optimizers[0] = torch.optim.Adam(self.parameters(), lr=self.lr)
            self.warmup_step = 0

        if self.param_loss_epochs is not None:
            if self.current_epoch >= self.param_loss_epochs:
                self.param_weight = 0.
                self.tuplewise_weight = 0.


        if self.current_epoch >= self.start_qm_epochs:
            # reset the loss weights to the values specified in the config:
            self.loss_fn.gradient_weight = float(self.gradient_weight)
            self.loss_fn.energy_weight = float(self.energy_weight)
            self.loss_fn.param_weight = float(self.param_weight)
            self.loss_fn.tuplewise_weight = float(self.tuplewise_weight)

            
        return super().on_train_epoch_start()



    def training_step(self, batch, batch_idx):

        self.assign_lr()

        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

        g, dsnames = batch

        g = self(g)

        loss = self.loss_fn(g, dsnames)

        if self.current_epoch > self.start_qm_epochs:
            self.log('losses/train_loss', loss, batch_size=self.batch_size(g), on_epoch=False, on_step=True)

            if self.current_epoch % self.log_train_interval == 0:
                with torch.no_grad():
                    self.train_evaluator.step(g, dsnames)

        return loss


    def validation_step(self, batch, batch_idx):
        if self.current_epoch < self.start_logging:
            return
        with torch.no_grad():
            g, dsnames = batch
            g = self(g)

            self.evaluator.step(g, dsnames)

            loss = self.loss_fn(g)
            self.log('losses/val_loss', loss, batch_size=self.batch_size(g), on_epoch=True)


    def on_validation_epoch_end(self):
        if self.current_epoch < self.start_logging:
            return
        metrics = self.evaluator.pool()
        for dsname in metrics.keys():
            for key in metrics[dsname].keys():
                self.log(f'{dsname}/val/{key}', metrics[dsname][key], on_epoch=True)

        # Calculate early stopping criterion as weighted sum of energy and gradient rmse of the individual datasets:
        gradient_avg = metrics["avg"]["rmse_gradients"]
        energy_avg = metrics["avg"]["rmse_energies"]
        early_stopping_loss = self.early_stopping_energy_weight * energy_avg + gradient_avg
        self.log('early_stopping_loss', early_stopping_loss, on_epoch=True)

        # Check whether the training should be stopped according to the finish criterion:
        elapsed_time = (time.time() - self.time_start + self.elapsed_time)/3600.
        relevant_finish_criterion = {k: v for k, v in self.finish_criterion.items() if k < elapsed_time}
        finish_criterion = min(relevant_finish_criterion.values()) if len(relevant_finish_criterion) > 0 else float("inf")
        if early_stopping_loss > finish_criterion:
            self.trainer.should_stop = True
            print(f"\nStopping training because early stopping criterion {early_stopping_loss} is larger than finish criterion {finish_criterion} after {elapsed_time} hours.\n", file=sys.stderr)


        if self.patience > 0:
            if self.current_epoch > self.start_qm_epochs:
                if early_stopping_loss < self.best_early_stopping_loss:
                    self.best_early_stopping_loss = float(early_stopping_loss)
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                if self.epochs_without_improvement > self.patience:
                    self.lr *= self.lr_decay
                    self.epochs_without_improvement = 0
                    self.best_early_stopping_loss = float(early_stopping_loss)


        # stop training if time limit is exceeded:
        if not self.time_limit is None:
            if time.time() - self.time_start + self.elapsed_time > self.time_limit*3600.:
                self.trainer.should_stop = True
                print(f"\nStopping training because time limit {self.time_limit} hours is exceeded.\n", file=sys.stderr)
        return super().on_validation_epoch_end()


    def test_step(self, batch, batch_idx):
        # turn off inference mode and clone data to enable grad computation for force calculation
        with torch.inference_mode(False):
            with torch.enable_grad():
                g, dsnames = batch
                g.nodes['n1'].data['xyz'] = g.nodes['n1'].data['xyz'].clone().detach()
            with torch.no_grad():
                g = self(g)
                self.test_evaluator.step(g, dsnames)

    def on_test_epoch_end(self):
        metrics = self.test_evaluator.pool(seed=42, n_bootstrap=1000)

        self.test_summary = metrics


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    
    def on_save_checkpoint(self, checkpoint):
        # Add elapsed time to the checkpoint
        checkpoint['elapsed_time'] = self.elapsed_time + time.time() - self.time_start
        checkpoint['lr'] = self.lr

    def on_load_checkpoint(self, checkpoint):
        # Load elapsed time from checkpoint. it is checked whether time.time() - start_time + elaped_time > time_limit in on_validation_epoch_end
        self.elapsed_time = checkpoint.get('elapsed_time', 0)
        self.start_time = time.time()
        try:
            self.lr = checkpoint.get('lr', self.lr)
        except Exception as e:
            print(f"Error in recovering the lr in on_load_checkpoint: {e}\nStarting with the lr in config file...", file=sys.stderr)
