#%%
# can we learn the hybridization feature with the graph model of grappa?
# if yes, can we a) include it in the loss for molecules where it is present? or b) use a nearest n transformer to learn it and concat it to the oither input feats? if yes, we could pretrain a small model that is always initialised with grappa to predict it only from atomic number and partial charge encoding.

from grappa.training.get_dataloaders import get_dataloaders
from grappa.models.graph_attention import GrappaGNN
import torch

#%%
# load a dataset:
train_loader, val_loader, test_loader = get_dataloaders(datasets=
    [
        "spice-des-monomers",
        "spice-dipeptide",
        "spice-pubchem",
        "gen2",
        "gen2-torsion",
        "pepconf-dlc",
        "protein-torsion",
        "rna-diverse",
    ],
    conf_strategy=1, train_batch_size=32, val_batch_size=256, test_batch_size=256)


#%%

#%%
class GraphModel(torch.nn.Module):
    def __init__(self, n_feats=32, n_classes=6):
        super().__init__()
        
        self.gnn = GrappaGNN(in_feat_name=['partial_charge', 'atomic_number', 'degree', 'ring_encoding'], out_feats=32, n_conv=0, n_att=1)
        self.lin_1 = torch.nn.Linear(32, 32)
        self.lin_out = torch.nn.Linear(32, n_classes)

    def forward(self, g):
        # Forward pass through the model
        g = self.gnn(g)
        scores = g.nodes['n1'].data['h']
        scores = torch.nn.functional.elu(scores)
        scores = self.lin_1(scores)
        scores = torch.nn.functional.elu(scores)
        scores = self.lin_out(scores)
        return scores

model = GraphModel()

#%%

def get_target(g):
    """
    Returns the target for the graph g in shape (n_atoms, 6)
    These are the one-hot encoded sp hybridization states we want to classify.
    """
    return g.nodes['n1'].data['sp_hybridization'].argmax(dim=1)

#%%
# train the model with lightning:

import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy

class LModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.accuracy = Accuracy(task="multiclass", num_classes=6)

    def training_step(self, batch, batch_idx):
        g, dsnames = batch
        target = get_target(g)
        out = self(g)
        loss = F.cross_entropy(out, target)
        acc = self.accuracy(out, target)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(dsnames))
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(dsnames))
        return loss

    def validation_step(self, batch, batch_idx):
        g, dsnames = batch
        target = get_target(g)
        out = self(g)
        loss = F.cross_entropy(out, target)
        acc = self.accuracy(out, target)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(dsnames))
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(dsnames))
        return loss

    def forward(self, g):
        return self.model(g)

    def configure_optimizers(self):
        # Configure optimizers and schedulers
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return val_loader

    def test_dataloader(self):
        return test_loader

# %%
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

# checkpoint callback:
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints',
    filename='best-model',
    save_top_k=1,
    mode='min',
    every_n_epochs=1,
)

wandb.init(project='Hybridization')

# Setup Wandb Logger
wandb_logger = WandbLogger(project='Hybridization')

# Instantiate the GraphModel
graph_model = LModel(model)

# Create a Trainer with Wandb Logger
trainer = Trainer(max_epochs=100, logger=wandb_logger, callbacks=[checkpoint_callback])

# Train the model
trainer.fit(graph_model)
# %%

# # test the checkpoint loading:
# # load the state dict:
# checkpoint_path = '/hits/fast/mbm/seutelf/grappa/tests/hybridization_feature/checkpoints/best-model-v2.ckpt'
# state_dict = torch.load(checkpoint_path)['state_dict']
# # remove the prefix:
# state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
# # load the model:
# model = GraphModel()
# model.load_state_dict(state_dict)

# #%%
# # validate that the model is pretrained:


# # Instantiate the GraphModel
# graph_model = LModel(model)

# # Create a Trainer with Wandb Logger
# trainer = Trainer(max_epochs=1)

# # Train the model
# trainer.fit(graph_model)
# # %%
