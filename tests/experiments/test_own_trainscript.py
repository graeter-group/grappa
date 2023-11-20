#%%
from grappa.data import Dataset, GraphDataLoader
from grappa.models import Energy, get_models
import torch
from pathlib import Path

#%%

model = get_models.get_full_model(in_feat_name=["atomic_number", "partial_charge", "ring_encoding"])

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

ds = Dataset.load(datapath/'spice-dipeptide')

split_ids = ds.calc_split_ids((0.8,0.1,0.1))

tr, vl, te = ds.split(*split_ids.values())
# %%
tr_loader, vl_loader, te_loader = (
    GraphDataLoader(s, batch_size=10, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    for s, shuffle, num_workers in ((tr, True, 1), (vl, False, 1), (te, False, 1))
)
# %%
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from grappa.utils.torch_utils import mean_absolute_error, root_mean_squared_error, invariant_mae, invariant_rmse
import numpy as np
from grappa.training.loss import ParameterLoss, TuplewiseEnergyLoss, EnergyLoss
from grappa import utils

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

tuplewise_energy_loss = TuplewiseEnergyLoss()

parameter_loss = ParameterLoss()

model = model.to('cuda')

#%%

for i in range(30):
    param_loss = 0
    tuple_en_loss = 0
    for g, dsname in tr_loader:
        g = g.to('cuda')
        g = model(g)
        p_loss = parameter_loss(g)
        e_loss = tuplewise_energy_loss(g)
        loss = 0
        # loss += p_loss
        loss += e_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        param_loss += p_loss.item()
        tuple_en_loss += e_loss.item()

    print(f"Epoch {i}:")
    print(f"train param loss:            {param_loss:.2f}")
    print(f"train tuplewise energy loss: {tuple_en_loss:.2f}")
    energies = []
    energies_ref = []
    energies_classical = []

    for g, dsname in vl_loader:
        with torch.no_grad():
            g = g.to('cuda')
            g = model(g)
            energies.append(utils.graph_utils.get_energies(g, suffix='', center=True).flatten())
            energies_ref.append(utils.graph_utils.get_energies(g, suffix='_ref').flatten())
            energies_classical.append(utils.graph_utils.get_energies(g, suffix='_classical_ff').flatten())

    energies = torch.cat(energies)
    energies_ref = torch.cat(energies_ref)
    energies_classical = torch.cat(energies_classical)

    print(f"val rmse energies:           {root_mean_squared_error(energies_ref, energies):.2f}")
    print(f"val rmse energies_classical: {root_mean_squared_error(energies_classical, energies):.2f}")

    print()

# %%

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

energy_loss = EnergyLoss()

for i in range(30):
    param_loss = 0
    tuple_en_loss = 0
    en_loss = 0
    for g, dsname in tr_loader:
        g = g.to('cuda')
        g = model(g)
        p_loss = parameter_loss(g)
        e_loss = tuplewise_energy_loss(g)
        qm_loss = energy_loss(g)
        loss = 0
        # loss += p_loss
        loss += 0.01 * e_loss
        loss += qm_loss/len(tr_loader)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        param_loss += p_loss.item()
        tuple_en_loss += e_loss.item()
        en_loss += qm_loss.item()

    print(f"Epoch {i}:")
    print(f"train param loss:            {param_loss:.2f}")
    print(f"train tuplewise energy loss: {tuple_en_loss:.2f}")
    print(f"train energy loss:           {en_loss:.2f}")
    energies = []
    energies_ref = []
    energies_classical = []

    for g, dsname in vl_loader:
        with torch.no_grad():
            g = g.to('cuda')
            g = model(g)
            energies.append(utils.graph_utils.get_energies(g, suffix='', center=True).flatten())
            energies_ref.append(utils.graph_utils.get_energies(g, suffix='_ref').flatten())
            energies_classical.append(utils.graph_utils.get_energies(g, suffix='_classical_ff').flatten())

    energies = torch.cat(energies)
    energies_ref = torch.cat(energies_ref)
    energies_classical = torch.cat(energies_classical)

    print(f"val rmse energies:           {root_mean_squared_error(energies_ref, energies):.2f}")
    print(f"val rmse energies_classical: {root_mean_squared_error(energies_classical, energies):.2f}")

    print()
# %%
import matplotlib.pyplot as plt

plt.scatter(energies_ref.cpu(), energies.cpu(), s=1)
plt.xlabel("ref")
plt.ylabel("grappa")
plt.show()

plt.scatter(energies_classical.cpu(), energies.cpu(), s=1)
plt.xlabel("classical")
plt.ylabel("grappa")
plt.show()