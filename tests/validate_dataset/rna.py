#%%
from grappa.data import Dataset, GraphDataLoader
import torch
from pathlib import Path
from grappa.utils import dgl_utils

# %%
dspath = Path(__file__).parents[2]/'data'/"dgl_datasets"

ds = Dataset.load(dspath/'rna-diverse')

# use openff 2.0.0 to predict forces and calculate their rmse wrt to the reference:

g, _ = ds[0]
print(g.nodes['n1'].data.keys())


from grappa.utils import openmm_utils, openff_utils
from grappa.models.energy import Energy
ref_writer = Energy(terms=['n2', 'n3', 'n4','n4_improper'], write_suffix="_classical", gradients=True, offset_torsion=True, suffix="_ref")


gradients_openff = []
gradients_ref = []

gradients_qm = []
full_grad_openff = []

for g, _ in ds:
    g = ref_writer(g)
    gradients_openff.append(g.nodes['n1'].data['gradient_classical'])
    gradients_ref.append(g.nodes['n1'].data['gradient_ref'])

    gradients_qm.append(g.nodes['n1'].data['gradient_qm'])
    full_grad_openff.append((g.nodes['n1'].data['gradient_openff-1.2.0']))


import torch

def rmse(predicted, actual):
    return torch.sqrt(torch.mean((predicted - actual) ** 2))

# Assuming gradients_openff, gradients_ref, and gradients_qm are lists of torch tensors
# with shapes [n_atoms, n_confs, 3] for each molecule

rmse_ref = []
rmse_full = []

for grad_openff, grad_ref, grad_qm, grad_full in zip(gradients_openff, gradients_ref, gradients_qm, full_grad_openff):
    # Calculate RMSE for OpenFF vs Reference
    rmse_ref.append(rmse(grad_openff, grad_ref))

    # Calculate RMSE for QM vs Reference
    rmse_full.append(rmse(grad_full, grad_qm))

# rmse_openff and rmse_qm now contain the RMSE values for each molecule
print('ref mean:', torch.tensor(rmse_ref).mean(), 'ref std:', torch.tensor(rmse_ref).std())

print('full mean:', torch.tensor(rmse_full).mean(), 'full std:', torch.tensor(rmse_full).std())
