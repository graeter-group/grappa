#%%
import torch
import numpy as np
from openmm.app import PDBFile, ForceField
from openmm.unit import angstrom
from grappa.utils.model_loading_utils import model_from_tag
from grappa.data import Molecule
from grappa.models.internal_coordinates import InternalCoordinates

#%%

"""
First, we need to construct a Molecule object from a PDB file
The Molecule requires bond information and nonboneded parameters. For this, we use openmm. It can also be constructed from a Gromacs topology file using kimmdy.
"""

PDBPATH = 'pdbfile.pdb'
#%%

openmm_topology = PDBFile(PDBPATH).getTopology()

openmm_system = ForceField('amber99sbildn.xml').createSystem(openmm_topology)

molecule = Molecule.from_openmm_system(openmm_system= openmm_system, openmm_topology=openmm_topology)

# %%
"""
This molecule object represents the molecular graph, the input to grappa. Now we transform it to a Deep Graph Library (DGL) graph since grappa uses DGL for its graph neural network.
Then, we load a grappa model with trained weights and feed to graph to the model to get the bond contributions.
"""

g = molecule.to_dgl()

model = model_from_tag('grappa-1.3')

g = model(g)

# %%
"""
The graph now contains the MM parameters.
Usually one would now convert the graph back to a format readable by openmm or kimmdy (including conversion to the correct units), however, we can also access them as numpy arrays directly.
Note that Grappa uses angstrom, kcal/mol, radian and derived units.
"""

# zero-based idxs of the atoms in the respective bond (corresponding to the order in the openmm topology, i.e. the order in the PDB file)
# shape (n_bonds, 2)
bond_idxs = g.nodes['n2'].data['idxs'].detach().cpu().numpy()

# bond lengths in angstrom
# shape (n_bonds,)
bond_eqs = g.nodes['n2'].data['eq'].detach().cpu().numpy()

# bond force constants in kcal/mol/angstrom^2
# shape (n_bonds,)
bond_ks = g.nodes['n2'].data['k'].detach().cpu().numpy()

# %%

"""
For calculating the bond energies, we also need to obtain the bond lengths, which can be done using the InternalCoordinates module. But first, we need to add 3D information (the positions of the atoms) to the 2D molecular graph.
"""
xyz_angstrom = PDBFile(PDBPATH).getPositions().value_in_unit(angstrom)

# construct torch tensor of shape (n_atoms, 3) and add batch dimension for conformations, which is expected by grappa
# shape (n_atoms, n_confs, 3)
xyz_angstrom = torch.tensor(xyz_angstrom).unsqueeze(1)

# write the 3D information to the graph (on the node level)
g.nodes['n1'].data['xyz'] = xyz_angstrom

#%%

calc_coords = InternalCoordinates()

# calculate bond distances (and angles and dihedrals)
g = calc_coords(g)

# obtain bond distances in angstrom
# shape (n_bonds, n_confs)
bond_distances = g.nodes['n2'].data['x'].detach().cpu().numpy()

# calculate bond energy contributions in kcal/mol:
# shape (n_bonds, n_confs)
bond_energies = 0.5 * bond_ks[:,None] * (bond_distances - bond_eqs[:,None])**2
print(f'bond_energies.shape: {bond_energies.shape}\n')
#%%

# energy of the bond between 0 and 1
# bond_idx = np.argwhere((bond_idxs == [0, 1]).all(axis=1)).item()
# print(f'Energy of bond between 0 and 1: {bond_energies[bond_idx]}')
# print(f'bond_idx: {bond_idx}\n')

total_energy = bond_energies.sum(axis=0)
print(f'total_energy.shape: {total_energy.shape}')
print(f'total_energy: {total_energy}')
# %%

############################################

#%%
"""
we can also directly compare to amber99sbildn bond energies by writing openmm parameters to the dgl graph:
"""

from grappa.data import Parameters

params = Parameters.from_openmm_system(openmm_system=openmm_system, mol=molecule)

g = params.write_to_dgl(g, suffix='_ref')

# repeat the procedure from above:

bond_eqs_amber = g.nodes['n2'].data['eq_ref'].detach().cpu().numpy()
bond_ks_amber = g.nodes['n2'].data['k_ref'].detach().cpu().numpy()

bond_energies_amber = 0.5 * bond_ks_amber[:,None] * (bond_distances - bond_eqs_amber[:,None])**2

total_energy_amber = bond_energies_amber.sum(axis=0)

print(f'Amber99 total_energy: {total_energy_amber}')
# print(f'Amber99 energy of bond between 0 and 1: {bond_energies_amber[bond_idx]}')

# %%
import matplotlib.pyplot as plt

plt.scatter(bond_energies_amber.flatten(), bond_energies.flatten())
plt.xlabel('Amber99 bond energies')
plt.ylabel('Grappa bond energies')
plt.title('Bond energy comparison')
plt.show()
# %%
plt.scatter(bond_eqs_amber.flatten(), bond_eqs.flatten())
plt.xlabel('Amber99 bond eqs')
plt.ylabel('Grappa bond eqs')
plt.title('Bond eq comparison')
plt.show()

# %%
plt.scatter(bond_ks_amber.flatten(), bond_ks.flatten())
plt.xlabel('Amber99 bond ks')
plt.ylabel('Grappa bond ks')
plt.title('Bond k comparison')
plt.show()
# %%