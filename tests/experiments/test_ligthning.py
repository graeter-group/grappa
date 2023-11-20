#%%
# NOTE: make update_configuration() a method of the LitModel class for learning rate, loss weights and so on

from grappa.data import Dataset, GraphDataLoader
from grappa.models import Energy, get_models, deploy
import torch
import json
from pathlib import Path
from grappa.utils.torch_utils import root_mean_squared_error, mean_absolute_error, invariant_rmse, invariant_mae
from grappa.training.evaluation import Evaluator, ExplicitEvaluator
#%%

# mit sp, ohne sp, ohne sp und deep

assert torch.cuda.is_available()

#%%

config = deploy.get_default_model_config('deep')

config['in_feat_name'] = ['atomic_number', 'partial_charge', 'ring_encoding']
# config['in_feat_name'] = ['atomic_number', 'partial_charge', 'ring_encoding', 'sp_hybridization']

model = deploy.model_from_config(config=config) # NOTE: the stat dict is not loaded here. make learnable?


class ParamFixer(torch.nn.Module):
    def forward(self, g):
        g.nodes['n2'].data['k'] = g.nodes['n2'].data['k'][:,0]
        g.nodes['n2'].data['eq'] = g.nodes['n2'].data['eq'][:,0]
        g.nodes['n3'].data['k'] = g.nodes['n3'].data['k'][:,0]
        g.nodes['n3'].data['eq'] = g.nodes['n3'].data['eq'][:,0]
        return g

model = torch.nn.Sequential(
    model,
    ParamFixer(),
    Energy(suffix=''),
    Energy(suffix='_ref', write_suffix="_classical_ff")
)
# %%
datapath = Path(__file__).parents[2]/'data'/"dgl_datasets"

ds = Dataset()
ds += Dataset.load(datapath/'spice-des-monomers')
ds += Dataset.load(datapath/'spice-dipeptide')
ds += Dataset.load(datapath/'spice-pubchem')
# ds += Dataset.load(datapath/'spice-dipeptide').shuffle().slice(None,300) # corresponds to [:300]
# ds += Dataset.load(datapath/'spice-pubchem').shuffle().slice(None,1000)
ds += Dataset.load(datapath/'spice-dipeptide')
ds += Dataset.load(datapath/'gen2')
ds += Dataset.load(datapath/'gen2-torsion')
ds += Dataset.load(datapath/'pepconf-dlc')
ds += Dataset.load(datapath/'protein-torsion')

ds.remove_uncommon_features()

split_ids = ds.calc_split_ids((0.8,0.1,0.1))

tr, vl, te = ds.split(*split_ids.values())
# %%
n_confs = 'mean'
n_mols = 10

tr_loader, vl_loader, te_loader = (
    GraphDataLoader(s, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers, pin_memory=True, conf_strategy=conf_strategy)
    for s, shuffle, num_workers, conf_strategy, batchsize in ((tr, True, 2, n_confs, n_mols), (vl, False, 2, 'all', 1), (te, False, 2, 'all', 1))
)


# dsweights = {'spice-des-monomers': 10, 'spice-pubchem':2}

# tr_loader, vl_loader, te_loader = (
#     GraphDataLoader(s, batch_size=10, shuffle=shuffle, num_workers=num_workers, pin_memory=True, weights=dsweights)
#     for s, shuffle, num_workers in ((tr, True, 2), (vl, False, 2), (te, False, 2))
# )



# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import numpy as np
from grappa.training.loss import ParameterLoss, TuplewiseEnergyLoss, EnergyLoss, GradientLoss, GradientLossMolwise, MolwiseLoss



#%%

class LitModel(pl.LightningModule):
    def __init__(self, model, tr_loader=None, vl_loader=None, te_loader=None):
        super().__init__()
        
        self.model = model
        self.tuplewise_energy_loss = TuplewiseEnergyLoss()
        
        self.loss = MolwiseLoss(gradient_weight=0, energy_weight=0, param_weight=1e-3)


        self.lrs = {0: 1e-4, 3: 1e-5, 150: 1e-6, 300: 1e-7} # which lr to use after which epoch
        self.lr = self.lrs[0]

        self.start_qm_epochs = 5

        self.restarts = [self.start_qm_epochs, 150, 350]
        self.warmup_steps = int(2e2) # the lr is linearly increased to self.lr in this number of steps
        self.warmup_step = None # for internal use
        self.warmup = True

        self.classical_epochs = 0
        self.energy_weight = 0.0
        self.gradient_weight = 10


        self.train_dataloader = lambda: tr_loader
        self.val_dataloader = lambda: vl_loader
        self.test_dataloader = lambda: te_loader

        self.log_train_interval = 5

        log_classical = False
        log_params = False

        self.evaluator = Evaluator(log_parameters=log_params, log_classical_values=log_classical)
        self.train_evaluator = Evaluator(log_parameters=log_params, log_classical_values=log_classical)

        self.check_eval = False # validate that the evaluation is consistent
        if self.check_eval:
            self.check_evaluator = ExplicitEvaluator(log_parameters=log_params, log_classical_values=log_classical, keep_data=False)
            

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
            self.loss.gradient_weight = self.gradient_weight
            self.loss.energy_weight = self.energy_weight
            

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

            if self.check_eval:
                self.check_evaluator.step(g, dsnames)

            loss = self.loss(g)
            if self.current_epoch > self.start_qm_epochs:
                self.log('losses/val_loss', loss, batch_size=self.batch_size(g), on_epoch=True)



    # At the end of the validation epoch, log squared energy and squared gradient, and reset class attributes
    def on_validation_epoch_end(self):

        metrics = self.evaluator.pool()
        for dsname in metrics.keys():
            for key in metrics[dsname].keys():
                if any([n in key for n in ["n2", "n3", "n4"]]):
                    self.log(f'parameters/{dsname}/val/{key}', metrics[dsname][key], on_epoch=True)
                else:
                    self.log(f'{dsname}/val/{key}', metrics[dsname][key], on_epoch=True)


        if self.check_eval:
            failed = False
            metrics2 = self.check_evaluator.pool()
            err_msg = ""
            for dsname in metrics2.keys():
                for key in metrics2[dsname].keys():
                    other_metric = metrics2[dsname][key]
                    this_metric = metrics[dsname][key]
                    if not np.isclose(other_metric, this_metric, atol=0.1):
                        failed = True
                        err_msg += f"\n {key}:  {other_metric} != {this_metric}"
            if failed:
                raise ValueError(f"Explicit evaluation failed: {err_msg}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lrs[0])
        return optimizer


#%%
# keep track of the model with best val loss but only after the first restart:
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='', # NOTE
    dirpath='checkpoints/',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min',
    save_last=True,
    every_n_epochs=20,
)
lit_model = LitModel(model, tr_loader, vl_loader, te_loader)
wandb_logger = WandbLogger()

#%%

trainer = pl.Trainer(logger=wandb_logger, gradient_clip_val=1e1, max_epochs=400, profiler="simple")#, callbacks=[checkpoint_callback])
trainer.fit(lit_model, tr_loader, vl_loader)
# %%
