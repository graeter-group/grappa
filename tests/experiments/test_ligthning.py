#%%
from grappa.data import Dataset, GraphDataLoader
from grappa.models import Energy, get_models
import torch
from pathlib import Path

#%%

model = get_models.get_full_model(in_feat_name=["atomic_number"])

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
    Energy.Energy(suffix=''),
    Energy.Energy(suffix='_ref', write_suffix="_classical_ff")
)
# %%
datapath = Path(__file__).parents[2]/'data'/"dgl_datasets"

ds = Dataset.load(datapath/'spice-dipeptide')

split_ids = ds.calc_split_ids((0.8,0.1,0.1))

tr, vl, te = ds.split(*split_ids.values())
# %%
tr_loader, vl_loader, te_loader = (
    GraphDataLoader(s, batch_size=10, shuffle=True, num_workers=1, pin_memory=True)
    for s in (tr, vl, te)
)
# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from grappa.utils.torch_utils import mean_absolute_error, root_mean_squared_error, invariant_mae, invariant_rmse
import numpy as np
from grappa.training.loss import ParameterLoss
from grappa import utils

class LitModel(pl.LightningModule):
    def __init__(self, model, tr_loader=None, vl_loader=None, te_loader=None):
        super().__init__()
        self.model = model
        self.loss_func = ParameterLoss()
        self.train_dataloader = lambda: tr_loader
        self.val_dataloader = lambda: vl_loader
        self.test_dataloader = lambda: te_loader

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        g, dsnames = batch
        g = self(g)
        loss = self.loss_func(g)
        self.log('train_loss', loss)

        # log energies and gradients
        with torch.no_grad():
            energies = utils.graph_utils.get_energies(g, suffix='')
            gradients = utils.graph_utils.get_gradients(g, suffix='')
            energies_ref = utils.graph_utils.get_energies(g, suffix='_ref')
            gradients_ref = utils.graph_utils.get_gradients(g, suffix='_ref')

            self.log('train_mae_energies', mean_absolute_error(energies_ref, energies).cpu())
            self.log('train_rmse_energies', root_mean_squared_error(energies_ref, energies).cpu())
            self.log('train_mae_gradients', invariant_mae(gradients_ref, gradients).cpu())
            self.log('train_rmse_gradients', invariant_rmse(gradients_ref, gradients).cpu())
            self.log('train_crmse_gradients', root_mean_squared_error(gradients_ref, gradients).cpu())

        return loss


    def validation_step(self, batch, batch_idx):
        g, dsname = batch
        g = self(g)
        with torch.no_grad():
            energies = utils.graph_utils.get_energies(g, suffix='')
            gradients = utils.graph_utils.get_gradients(g, suffix='')
            energies_ref = utils.graph_utils.get_energies(g, suffix='_ref')
            gradients_ref = utils.graph_utils.get_gradients(g, suffix='_ref')
            if batch_idx == 0:
                self.val_energies = energies
                self.val_gradients = gradients
                self.val_energies_ref = energies_ref
                self.val_gradients_ref = gradients_ref
            else:
                self.val_energies = torch.cat((self.val_energies, energies), dim=0)


            if batch_idx == len(self.val_dataloader()) - 1:
                self.log('val_mae_energies', mean_absolute_error(self.val_energies_ref, self.val_energies))
                self.log('val_rmse_energies', np.sqrt(mean_squared_error(self.val_energies_ref, self.val_energies)))
                self.log('val_mae_gradients', mean_absolute_error(self.val_gradients_ref, self.val_gradients))
                self.log('val_rmse_gradients', np.sqrt(mean_squared_error(self.val_gradients_ref, self.val_gradients)))


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

#%%
lit_model = LitModel(model, tr_loader, vl_loader, te_loader)
wandb_logger = WandbLogger()

#%%

trainer = pl.Trainer(logger=wandb_logger)
trainer.fit(lit_model, tr_loader, vl_loader)
# %%
