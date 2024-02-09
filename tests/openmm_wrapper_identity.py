#%%

# test whether the openmm wrapper works as expected by implementing the identity and comparing the results.

from grappa.wrappers.openmm_wrapper import openmm_Grappa
from openmm.app import PDBFile, ForceField

#%%
#####################
pdb = PDBFile('../examples/usage/T4.pdb')
topology = pdb.topology
ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
system = ff.createSystem(pdb.topology)
#####################


#####################
# # molecule with propers but no improper torsions:
# smiles = 'ON(=O)'

# from grappa.utils.openff_utils import get_openmm_system
# system, topology, openff_mol = get_openmm_system(mapped_smiles=None, smiles=smiles, partial_charges=0)

# # get positions:
# openff_mol.generate_conformers(n_conformers=1)
# from openff.units import unit as openff_unit
# positions = openff_mol.conformers[0]/openff_unit.angstrom
#####################



import copy

original_system = copy.deepcopy(system)

#%%
# create a graph that stores reference parameters:
from grappa.data import Molecule, Parameters
mol = Molecule.from_openmm_system(openmm_system=system, openmm_topology=topology)
params = Parameters.from_openmm_system(openmm_system=system, mol=mol)
g = mol.to_dgl()
g = params.write_to_dgl(g)

#%%
import torch
from grappa.constants import BONDED_CONTRIBUTIONS

class identity(torch.nn.Module):
    """
    Model that simply writes the parameters with suffix from some graph in the given graph.
    """
    def forward(self, graph):
        suffix = '_ref'
        for lvl, param in BONDED_CONTRIBUTIONS:
            graph.nodes[lvl].data[param] = g.nodes[lvl].data[param+suffix]
        return graph

model = identity()
#%%

# build a grappa model that handles the ML pipeline
grappa = openmm_Grappa(model, device='cpu')

# write grappa parameters to the system:
system = grappa.parametrize_system(system, topology)
# %%
from openmm.unit import angstrom, kilocalorie_per_mole
from grappa.utils.openmm_utils import get_energies
import numpy as np

positions = pdb.positions.value_in_unit(angstrom)
positions = np.array([positions])

#%%
# remove the torsion force from the system

# from grappa.utils.openmm_utils import remove_forces_from_system
# system = remove_forces_from_system(system, 'PeriodicTorsionForce')
# original_system = remove_forces_from_system(original_system, 'PeriodicTorsionForce')
#%%
en, grads = get_energies(system, positions)
orig_en, orig_grads = get_energies(original_system, positions)

assert np.allclose(grads, orig_grads, atol=1e-3)

# %%
print('original energy:', orig_en)
print('grappa_identity energy:', en)
# %%
import matplotlib.pyplot as plt

plt.scatter(orig_grads.flatten(), grads.flatten())
plt.xlabel('original gradients')
plt.ylabel('grappa_identity gradients')
plt.show()

# %%
