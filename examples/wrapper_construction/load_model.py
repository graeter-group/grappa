#%%
from grappa.utils.model_loading_utils import model_from_tag
model = model_from_tag('latest')

# %%
# validate that the models predictions are somewhat reasonable:
from grappa.data import Dataset
from grappa.models.energy import Energy
from matplotlib import pyplot as plt
import numpy as np
import torch

ds = Dataset.from_tag('tripeptides_amber99sbildn')

energy_writer = Energy()

energy_grappa = []
energy_qm = []
energy_classical = []

for g, dsname in ds:

    with torch.no_grad():
        g = model(g)
        g = energy_writer(g)

    energy = g.nodes['g'].data['energy'].detach().numpy()
    energy -= energy.mean()

    energy_ref = g.nodes['g'].data['energy_ref'].detach().numpy()
    energy_ref -= energy_ref.mean()

    energy_classical_ff = g.nodes['g'].data['energy_reference_ff'].detach().numpy()
    energy_classical_ff -= energy_classical_ff.mean()

    energy_grappa.append(energy.flatten())
    energy_qm.append(energy_ref.flatten())
    energy_classical.append(energy_classical_ff.flatten())

energy_grappa = np.concatenate(energy_grappa, axis=0)
energy_qm = np.concatenate(energy_qm, axis=0)
energy_classical = np.concatenate(energy_classical, axis=0)

plt.scatter(energy_qm, energy_grappa, label='Grappa')
plt.scatter(energy_qm, energy_classical, label='Amber99sb')
plt.plot(energy_qm, energy_qm, color='black')

plt.xlabel('qm energy')
plt.ylabel('predicted energy')

plt.show()
# %%
