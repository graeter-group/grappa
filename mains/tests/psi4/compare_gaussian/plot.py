#%%
from single_points import calc_states
from pathlib import Path


calc_states(Path(__file__).parent, n_states=None)

#%%

# plot the energies and forces:
import numpy as np
import matplotlib.pyplot as plt

openmm_energies = np.load("energies.npy")
openmm_forces = np.load("forces.npy")
psi4_energies = np.load("psi4_energies.npy")
psi4_forces = np.load("psi4_forces.npy")

# Subtract the mean of the energy per molecule
psi4_energies -= np.mean(psi4_energies)
openmm_energies -= np.mean(openmm_energies)

openmm_energies = openmm_energies[:len(psi4_energies)].flatten()
openmm_forces = openmm_forces[:len(psi4_forces)].flatten()

psi4_energies = psi4_energies.flatten()
psi4_forces = psi4_forces.flatten()


# Plot the data
fig, ax = plt.subplots(1,2, figsize=(10,5))

ax[0].scatter(psi4_energies, openmm_energies)
ax[0].plot(psi4_energies, psi4_energies, color='black')
ax[0].set_xlabel('psi4 energy [kcal/mol]')
ax[0].set_ylabel('gaussian energy [kcal/mol]')

ax[1].scatter(psi4_forces, openmm_forces)
ax[1].plot(psi4_forces, psi4_forces, color='black')
ax[1].set_xlabel('psi4 force [kcal/mol/Å]')
ax[1].set_ylabel('gaussian force [kcal/mol/Å]')


rmse_energies = np.sqrt(np.mean((psi4_energies - openmm_energies)**2))

rmse_forces = np.sqrt(np.mean((psi4_forces - openmm_forces)**2))

ax[0].text(0.05, 0.95, f"RMSE: {rmse_energies:.2f} kcal/mol", transform=ax[0].transAxes, verticalalignment='top')


ax[1].text(0.05, 0.95, f"RMSE: {rmse_forces:.2f} kcal/mol/Å", transform=ax[1].transAxes, verticalalignment='top')

plt.tight_layout()


plt.savefig(f'summary.png')
