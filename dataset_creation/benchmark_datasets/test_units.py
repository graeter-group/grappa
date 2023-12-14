#%%
import dgl
from pathlib import Path
import torch
import openmm
import openff.toolkit
import json

#%%
dspath = Path("/hits/fast/mbm/seutelf/data/spice-dipeptide")
[g], labels = dgl.load_graphs(str(dspath / "1/heterograph.bin"))

#%%
import io
with open(str(dspath / "1/mol.json"), 'r') as file:
    moldata = json.load(file)
    # convert from str to dict:
    moldata = json.loads(moldata)
if not 'pertial_charge_unit' in moldata.keys():
    moldata['partial_charge_unit'] = moldata['partial_charges_unit']
if "hierarchy_schemes" not in moldata.keys():
    moldata["hierarchy_schemes"] = dict()
#%%
mol = openff.toolkit.topology.Molecule.from_dict(moldata)
mol.partial_charges
#%%
# %%
xyz = g.nodes['n1'].data['xyz'].transpose(0,1).numpy()
grads = g.nodes['n1'].data['u_qm_prime'].transpose(0,1).numpy()

# %%
import numpy as np
from openmm.unit import Quantity

from openmm import XmlSerializer, Platform, Context, VerletIntegrator

from openmm.unit import mole, hartree, bohr, angstrom, kilocalories_per_mole, nanometer, kilojoule_per_mole

ENERGY_UNIT = kilocalories_per_mole
DISTANCE_UNIT = angstrom
FORCE_UNIT = ENERGY_UNIT / DISTANCE_UNIT

PARTICLE = mole.create_unit(6.02214076e23 ** -1, "particle", "particle")
HARTREE_PER_PARTICLE = hartree / PARTICLE

qca_force = HARTREE_PER_PARTICLE / bohr

def compute_energies(system, positions):
    if isinstance(positions, list):
        positions = np.array(positions)

    assert len(positions.shape) == 3

    platform = Platform.getPlatformByName("Reference")
    integrator = VerletIntegrator(0.001)
    context = Context(system, integrator, platform)
    energies = []
    forces = []
    for pos in positions:
        context.setPositions(pos)
        state = context.getState(getEnergy=True, getForces=True)
        energies.append(state.getPotentialEnergy().value_in_unit(ENERGY_UNIT))
        forces.append(state.getForces(asNumpy=True).value_in_unit(FORCE_UNIT))

    return np.array(energies), np.array(forces)

#%%
from openff.toolkit import Topology, ForceField
forcefield_version = 'openff-2.0.0'

topology = Topology.from_molecules(mol)
forcefield = ForceField(forcefield_version+'.offxml')
openmm_system = forcefield.create_openmm_system(topology, charge_from_molecules=[mol])
system = openmm_system

#%%

esp_distance = bohr
esp_force = HARTREE_PER_PARTICLE / bohr
esp_energy = HARTREE_PER_PARTICLE

pos = Quantity(xyz, unit=esp_distance).value_in_unit(nanometer)
grad_in_unit = Quantity(grads, unit=qca_force).value_in_unit(FORCE_UNIT)

grad_in_unit = grad_in_unit

energies, forces = compute_energies(system, pos)

assert grad_in_unit.shape == forces.shape, f"{grad_in_unit.shape} != {forces.shape}"

#%%
import matplotlib.pyplot as plt

# np.random.shuffle(grad_in_unit)
plt.scatter(-grad_in_unit.flatten(), forces.flatten())
# %%
u_qm = g.nodes['g'].data['u_qm'].numpy()
en_in_unit = Quantity(u_qm, unit=esp_energy).value_in_unit(ENERGY_UNIT)

en_in_unit -= en_in_unit.mean()
energies -= energies.mean()

plt.scatter(energies, en_in_unit)
# %%
# UNITS FIT, esp units seem to be HARTREE_PER_PARTICLE and bohr

def get_partial_charges(system):

    nonb_force = None
    for f in system.getForces():
        if isinstance(f, openmm.NonbondedForce):
            nonb_force = f
            break

    partial_charges = np.array([nonb_force.getParticleParameters(i)[0].value_in_unit(openmm.unit.elementary_charge) for i in range(nonb_force.getNumParticles())])

    return partial_charges

openmm_system = forcefield.create_openmm_system(topology, charge_from_molecules=[mol])
am1bc_charges = get_partial_charges(openmm_system)
am1bc_charges
#%%
openmm_system = forcefield.create_openmm_system(topology)
openff2_charges = get_partial_charges(openmm_system)
openff2_charges

# %%
# the charges are slightly different!