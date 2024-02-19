from grappa.training.loss import MolwiseLoss
from grappa.training.evaluation import FastEvaluator
import json

import torch
import pytorch_lightning as pl
from typing import List, Dict
import time
import sys
import traceback


class LitModel(pl.LightningModule):
    def __init__(self, model, tr_loader, vl_loader, te_loader, 
                 lr=1e-4, 
                 start_qm_epochs=1, add_restarts=[],
                 warmup_steps=int(2e2),
                 energy_weight=1., gradient_weight=1e-1, tuplewise_weight=0,
                 param_weight=1e-4, proper_regularisation=1e-5, improper_regularisation=1e-5,
                 log_train_interval=5, log_classical=False, log_params=False, weight_decay=0.,
                 early_stopping_energy_weight=2.,
                 log_metrics=True,
                 patience:int=30, lr_decay:float=0.8, time_limit:float=None, finish_criterion:Dict[int, float]={}, param_loss_epochs:int=None, param_weights_by_dataset:Dict[str,float]={}):
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
            patience (int): Number of epochs without increase of the early stopping criterion (i.e. the weighted sum of energy/force rmse averaged over the dataset types) after which the learning rate is decreased: The best early stopping criterion value is stored and if the current value is larger than the best value, the number of epochs without improvement is increased by 1, else set to zero. If the number of epochs without improvement is larger than the patience, the learning rate is decreased by a factor of decay, the number of epochs without improvement is set to zero and the best early stopping criterion value is updated to the current value.
            lr_decay (float): Factor by which to decrease the learning rate if the early stopping criterion does not improve for patience epochs.
            time_limit (float): Time limit in hours. If the training takes longer than this, the training is stopped.
            finish_criterion (dict): Dictionary mapping from time in hours to maximum early stopping criterion value. If the early stopping criterion is larger than the maximum value for the given time, the training is stopped. This is useful for sweep runs, where the training should be stopped if it is not promising.
            param_loss_epochs (int): Epoch number from which the parameter and tuplewise loss weights are set to zero and the optimizer is restarted. If None, has no effect. Default: None.
            param_weights_by_dataset (dict): Dictionary mapping from dataset name to weight of the parameter loss for this dataset. This overwrites the value of param_weight for entries of the datasets occuring in the dictionary. Default: {}.
        """
        super().__init__()
        
        self.model = model
        
        # first, set energy and gradient weight to zero to only train the parameters. these are re-setted in on_train_epoch_start
        self.loss = MolwiseLoss(gradient_weight=0, energy_weight=0, param_weight=1e-3, tuplewise_weight=tuplewise_weight, proper_regularisation=proper_regularisation, improper_regularisation=improper_regularisation, param_weights_by_dataset=param_weights_by_dataset) if start_qm_epochs > 0 else MolwiseLoss(gradient_weight=gradient_weight, energy_weight=energy_weight, param_weight=param_weight, tuplewise_weight=tuplewise_weight, proper_regularisation=proper_regularisation, improper_regularisation=improper_regularisation, param_weights_by_dataset=param_weights_by_dataset)

        self.lr = lr
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

        self.param_loss_epochs = param_loss_epochs
        if not self.param_loss_epochs is None:
            self.restarts = sorted(list(set(self.restarts + [self.param_loss_epochs])))

        self.train_dataloader = lambda: tr_loader
        self.val_dataloader = lambda: vl_loader
        self.test_dataloader = lambda: te_loader

        self.log_train_interval = log_train_interval

        self.finish_criterion = finish_criterion

        self.evaluator = FastEvaluator(log_parameters=log_params, log_classical_values=log_classical)
        self.train_evaluator = FastEvaluator(log_parameters=log_params, log_classical_values=log_classical)

        self.early_stopping_energy_weight = early_stopping_energy_weight

        self.log_metrics = log_metrics

        self.time_limit = time_limit

        self.elapsed_time = 0 # time from a previous run, if the training is restarted.  it is checked whether time.time() - start_time + elaped_time > time_limit in on_validation_epoch_end


        # helper variables:
        self.best_early_stopping_loss = float("inf")
        self.epochs_without_improvement = 0

        self.time_start = time.time()


        self.val_failed = False
        self.val_fail_counter = 0


    def forward(self, x):
        return self.model(x)

    def batch_size(self, g):
        return g.num_nodes('g')


    def get_lr(self):
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


    def set_lr(self):
        """
        Sets the learning rate of the optimizer to self.lr. Resets the learning rate (self.lr) and the optimizer after restarts. To be called once per train epoch.
        """

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
            if self.current_epoch % self.log_train_interval == 0 and self.current_epoch > self.start_qm_epochs:
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

        if self.param_loss_epochs is not None:
            if self.current_epoch >= self.param_loss_epochs:
                self.param_weight = 0.
                self.tuplewise_weight = 0.


        if self.current_epoch >= self.start_qm_epochs:
            # reset the loss weights to the values specified in the config:
            self.loss.gradient_weight = float(self.gradient_weight)
            self.loss.energy_weight = float(self.energy_weight)
            self.loss.param_weight = float(self.param_weight)
            self.loss.tuplewise_weight = float(self.tuplewise_weight)

            
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

        loss = self.loss(g, dsnames)

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
        if not self.val_failed:
            if self.log_metrics:
                metrics = self.evaluator.pool()
                for dsname in metrics.keys():
                    for key in metrics[dsname].keys():
                        if any([n in key for n in ["n2", "n3", "n4"]]):
                            self.log(f'parameters/{dsname}/val/{key}', metrics[dsname][key], on_epoch=True)
                        elif self.current_epoch > self.start_qm_epochs:
                            self.log(f'{dsname}/val/{key}', metrics[dsname][key], on_epoch=True)

                if self.current_epoch > self.start_qm_epochs:
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
                assert self.log_metrics, "Early stopping criterion is only implemented if metrics are logged."

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