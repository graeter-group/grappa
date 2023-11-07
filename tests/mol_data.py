#%%
from pathlib import Path
import numpy as np

dspath = Path(__file__).parents[1]/'data'/"datasets"/"spice-des-monomers"

data = np.load(dspath/"0.npz")

smiles = data['mapped_smiles'].item()
print([k for k in data.keys()])
energy = data['energy_qm']
gradient = data['gradient_qm']
xyz = data['xyz']
charges = data['am1bcc_elf_charges']

energy_ref = data['energy_ref']
gradient_ref = data['gradient_ref']

# %%

from grappa.data import MolData

mol = MolData.from_smiles(mapped_smiles=smiles, xyz=xyz, energy=energy, gradient=gradient, openff_forcefield='openff_unconstrained-1.2.0.offxml', partial_charges=charges, energy_ref=energy_ref, gradient_ref=gradient_ref)
# %%
from grappa.utils import openmm_utils, openff_utils

system, topol = openff_utils.get_openmm_system(smiles, openff_forcefield='openff_unconstrained-1.2.0.offxml', partial_charges=charges)

energies_cl, forces_cl = openmm_utils.get_energies(openmm_system=system, xyz=xyz)
# %%
print("compare classical ff with qm:")
import matplotlib.pyplot as plt
plt.scatter(mol.gradient, -forces_cl)
plt.show()
mol.energy -= mol.energy.mean()
energies_cl -= energies_cl.mean()
plt.scatter(mol.energy, energies_cl)
plt.show()
# %%
print("validate that the reference energy is qm - nonbonded with the given charges:")
system = openmm_utils.remove_forces_from_system(system, keep=['NonbondedForce'])
nonb_energies, nonb_forces = openmm_utils.get_energies(openmm_system=system, xyz=xyz)

u_reference = mol.energy - nonb_energies
u_reference -= u_reference.mean()

mol.reference_energy -= mol.reference_energy.mean()

plt.scatter(u_reference, mol.reference_energy)
plt.show()
print("rmse:", np.sqrt(np.mean((u_reference - mol.reference_energy)**2)), "kcal/mol")
# %%
# same for reference gradient
grad_reference = mol.gradient - nonb_forces
plt.scatter(grad_reference, mol.reference_gradient)
plt.show()
print("rmse:", np.sqrt(np.mean((grad_reference - mol.reference_gradient)**2)), "kcal/mol/A")
#%%
g = mol.to_dgl()
g.nodes['n1'].data['atomic_number'].shape
# %%
from grappa.data import Parameters
params = Parameters.from_dgl(g, suffix='_ref')
assert np.allclose(params.bond_eq, mol.classical_parameters.bond_eq, atol=1e-3)
# %%
# now predict bonded energy of the classical ff:
system, topol = openff_utils.get_openmm_system(smiles, openff_forcefield='openff_unconstrained-1.2.0.offxml', partial_charges=charges)
system = openmm_utils.remove_forces_from_system(system, exclude=['NonbondedForce'])
b_energies, b_forces = openmm_utils.get_energies(openmm_system=system, xyz=xyz)
# %%
from grappa.models.energy import WriteEnergy
from grappa.models.geometry import GeometryInGraph

geom = GeometryInGraph()

g = geom(g)

writer = WriteEnergy(suffix='_ref', write_suffix='_val')
g = writer(g)
grappa_bonded_e = g.nodes['g'].data['energy_val'].flatten().numpy()
grappa_bonded_e -= grappa_bonded_e.mean()

b_energies = b_energies.flatten()
b_energies -= b_energies.mean()

plt.scatter(grappa_bonded_e, b_energies)
plt.show()

print('rmse between grappa and openmm bonded energy:\n', np.sqrt(np.mean((grappa_bonded_e - b_energies)**2)), 'kcal/mol')
# %%
