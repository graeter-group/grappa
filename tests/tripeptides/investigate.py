#%%
from grappa.training.get_dataloaders import get_dataloaders
#%%
tr,vl,te = get_dataloaders(['tripeptides_amber99sbildn'])
# %%
import matplotlib.pyplot as plt
import numpy as np

grads = []
energies = []

energy_reference_ff = []
grad_reference_ff = []

for g, dsnames in tr:
    grads.append(g.nodes['n1'].data['gradient_ref'].numpy().flatten())
    energies.append(g.nodes['g'].data['energy_ref'].numpy().flatten())

    ref_ff_energy = g.nodes['g'].data['energy_reference_ff'].numpy()
    ref_ff_energy -= ref_ff_energy.mean()
    energy_reference_ff.append(ref_ff_energy.flatten())

    grad_reference_ff.append(g.nodes['n1'].data['gradient_reference_ff'].numpy().flatten())

grads = np.concatenate(grads)

plt.hist(grads, bins=100)
plt.show()

energies = np.concatenate(energies)

plt.hist(energies, bins=10)
plt.show()

energy_reference_ff = np.concatenate(energy_reference_ff)
plt.scatter(energies, energy_reference_ff)
plt.show()

grad_reference_ff = np.concatenate(grad_reference_ff)
plt.scatter(grads, grad_reference_ff)
plt.show()
# %%