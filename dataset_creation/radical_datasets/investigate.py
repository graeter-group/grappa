"""
Script showing that either there is an error in the openmm force field amber99sbildn or in the QM calculation of the negatively charged molecule. Remove skipping of E and D in the dataset creation and run this script to see that the wrong gradients originate from the part with negative charge.
"""
#%%
from grappa.data import Dataset
import matplotlib.pyplot as plt
import torch

#%%
dstag = 'capped_peptide_amber99sbildn'

ds = Dataset.from_tag(dstag)
# %%
grads_qm = []
grads_amber = []
mol_ids = []
for i, mol in enumerate(ds):
    g, _ = mol
    mol_ids.append(ds.mol_ids[i])
    grad_qm = g.nodes['n1'].data['gradient_qm']
    grad_amber = g.nodes['n1'].data['gradient_amber99sbildn']
    grads_qm.append(grad_qm.flatten())
    grads_amber.append(grad_amber.flatten())

# %%
rmses = []
for grad_qm, grad_amber in zip(grads_qm, grads_amber):
    rmse = torch.sqrt(torch.mean((grad_qm-grad_amber)**2)).item()
    rmses.append(rmse)


# %%
# bar plot:
plt.figure(figsize=(12,6))
plt.bar(mol_ids, rmses)
plt.xticks(rotation=90)
plt.ylabel('RMSE')
plt.xlabel('Molecule ID')
plt.title('RMSE between QM and Amber99sbildn gradients')

# %%
# find the molecule E_standard:
idx = ds.mol_ids.index('E_standard')

g = ds[idx][0]

g.nodes['n1'].data['partial_charge'].sum()
# %%
grad_var = (g.nodes['n1'].data['gradient_amber99sbildn']**2).sum(dim=-1).mean(dim=-1).tolist()
partial_charge = g.nodes['n1'].data['partial_charge'].tolist()

import numpy as np
# two bar plot with those two quantities:
fig, ax = plt.subplots(2,1,figsize=(12,6))
ax[0].bar(np.arange(len(grad_var)), grad_var, label='Gradient variance')

ax[1].bar(np.arange(len(partial_charge)), partial_charge, label='Partial charge')

# %%
