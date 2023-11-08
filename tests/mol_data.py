#%%
from pathlib import Path
import numpy as np

dspath = Path(__file__).parents[1]/'data'/"datasets"/"spice-des-monomers"

data = np.load(dspath/"34.npz")

smiles = data['mapped_smiles'].item()
# print([k for k in data.keys()])
energy = data['energy_qm']
gradient = data['gradient_qm']
xyz = data['xyz']
charges = data['am1bcc_elf_charges']

energy_ref = data['energy_ref']
gradient_ref = data['gradient_ref']

# print(np.mean(np.square(energy_ref - energy)))
# %%

from grappa.data import MolData

mol = MolData.from_smiles(mapped_smiles=smiles, xyz=xyz, energy=energy, gradient=gradient, openff_forcefield='openff_unconstrained-2.0.0.offxml', partial_charges=charges, energy_ref=energy_ref, gradient_ref=gradient_ref)
# %%
from grappa.utils import openmm_utils, openff_utils

system, topol, _ = openff_utils.get_openmm_system(smiles, openff_forcefield='openff_unconstrained-2.0.0.offxml', partial_charges=charges)

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
assert 1e-10 < np.sqrt(np.mean((energies_cl - mol.energy)**2)) < 10
# %%
print("validate that the reference energy is qm - nonbonded with the given charges:")
system = openmm_utils.remove_forces_from_system(system, keep=['NonbondedForce'])
nonb_energies, nonb_forces = openmm_utils.get_energies(openmm_system=system, xyz=xyz)

u_reference = mol.energy - nonb_energies
u_reference -= u_reference.mean()

mol.energy_ref -= mol.energy_ref.mean()

plt.scatter(u_reference, mol.energy_ref)
plt.show()
print("rmse:", np.sqrt(np.mean((u_reference - mol.energy_ref)**2)), "kcal/mol")
assert np.sqrt(np.mean((u_reference - mol.energy_ref)**2)) < 1e-1
# %%
# same for reference gradient
grad_reference = mol.gradient + nonb_forces
plt.scatter(grad_reference, mol.gradient_ref)
plt.show()
print("rmse:", np.sqrt(np.mean((grad_reference - mol.gradient_ref)**2)), "kcal/mol/A")
assert np.sqrt(np.mean((grad_reference - mol.gradient_ref)**2)) < 1e-1
#%%
g = mol.to_dgl()
g.nodes['n1'].data['atomic_number'].shape
# %%
from grappa.data import Parameters
params = Parameters.from_dgl(g, suffix='_ref')
assert np.allclose(params.bond_eq, mol.classical_parameters.bond_eq, atol=1e-3)
# %%
# now predict bonded energy of the classical ff:
system, topol, _ = openff_utils.get_openmm_system(smiles, openff_forcefield='openff_unconstrained-2.0.0.offxml', partial_charges=charges)
system = openmm_utils.remove_forces_from_system(system, remove=['NonbondedForce'])
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
assert np.sqrt(np.mean((grappa_bonded_e - b_energies)**2)) < 1e-3
# %%

# do the nonbonded energies from openff and openff_unconstrained match?
system, topol, _ = openff_utils.get_openmm_system(smiles, openff_forcefield='openff-2.0.0.offxml', partial_charges=charges)
system = openmm_utils.remove_forces_from_system(system, keep=['NonbondedForce'])

nonb_energies, nonb_forces = openmm_utils.get_energies(openmm_system=system, xyz=xyz)

system, topol, _ = openff_utils.get_openmm_system(smiles, openff_forcefield='openff_unconstrained-2.0.0.offxml', partial_charges=charges)
system = openmm_utils.remove_forces_from_system(system, keep=['NonbondedForce'])

nonb_energies2, nonb_forces2 = openmm_utils.get_energies(openmm_system=system, xyz=xyz)
# %%
print("rmse between unconstrained and constrained nonbonded energies:\n", np.sqrt(np.mean((nonb_energies - nonb_energies2)**2)), 'kcal/mol')
print("rmse between unconstrained and constrained nonbonded forces:\n", np.sqrt(np.mean((nonb_forces - nonb_forces2)**2)), 'kcal/mol/A')
assert np.sqrt(np.mean((nonb_energies - nonb_energies2)**2)) < 1e-3
assert np.sqrt(np.mean((nonb_forces - nonb_forces2)**2)) < 1e-3
# %%
assert np.all(mol.improper_energy_ref == 0)

# %%
# now for a molecule with impropers:
from grappa.data import MolData
smiles = 'CC(C)(C)C(=O)OC1CC2C3CCC4=CC(=O)CCC4=C3C(CCC12C)C(=O)OC(C)(C)C'
from openff.toolkit import Molecule as OpenFFMolecule
from openmm import unit

openff_mol = OpenFFMolecule.from_smiles(smiles, allow_undefined_stereo=True)
openff_mol.generate_conformers(n_conformers=1)


mapped_smiles = openff_mol.to_smiles(mapped=True)

xyz = np.array([openff_mol.conformers[i].to_openmm().value_in_unit(unit.angstrom) for i in range(1)])
energies = np.zeros(1)
gradient = np.zeros((1, len(openff_mol.atoms), 3))
charges = np.zeros(len(openff_mol.atoms))
energy_ref = np.zeros(1)
gradient_ref = np.zeros((1, len(openff_mol.atoms), 3))

moldata = MolData.from_smiles(mapped_smiles=mapped_smiles, xyz=xyz, energy=energies, gradient=gradient, openff_forcefield='openff_unconstrained-2.0.0.offxml', partial_charges=charges, energy_ref=energy_ref, gradient_ref=gradient_ref)
# %%
assert moldata.improper_gradient_ref.std() > 1e-5
# %%
