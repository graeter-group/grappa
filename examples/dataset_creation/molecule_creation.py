"""
CONSTRUCTION BY LISTS

To initialize a molecule, one needs to specify:
    - A list of unique atom ids (integers) [The ids must not be indices: They do not represent a position in a list but just serve as identifier. This enables to use indexes of larger topologies as atom ids if one wishes to construct a sub-molecule.]
    - A list of bonds (tuples of two atom ids)
    - A list of improper torsions (tuples of four atom ids) [The order of the atoms in the tuples does not matter, grappa will sort them and identify central atoms automatically. These can not be inferred from the molecular graph since not all possible improper torsions are actually used in common force fields.]
    - A list of atomic numbers (integers) in the same order as the atom ids
    - A list of partial charges in the same order as the atom ids
    Lets take the example of methane:
"""

#%%
from grappa.data import Molecule

#%%
atoms = [1, 2, 3, 4, 5]
bonds = [(1, 2), (1, 3), (1, 4), (1, 5)]
impropers = []
partial_charges = [-0.4, 0.1, 0.1, 0.1, 0.1]
atomic_numbers = [6, 1, 1, 1, 1]

methane = Molecule(atoms=atoms, bonds=bonds, impropers=impropers, partial_charges=partial_charges, atomic_numbers=atomic_numbers)

print(methane) # informative string representation
# %%
# Now the molecule has angles and propers:
print(methane.angles)
print(methane.propers)
# %%

# now lets construct the same molecule but with an improper torsion:
impropers = [(1,2,3,4)]

methane = Molecule(atoms=atoms, bonds=bonds, impropers=impropers, partial_charges=partial_charges, atomic_numbers=atomic_numbers)

# grappa creates three impropers for each set of atoms that is added: The central atom is at position grappa.constants.IMPROPER_CENTRAL_IDX
print(methane.impropers)
# %%
# if the improper torsion that we provided is not actually an improper torsion (i.e. there is no atom to which the three remaining atoms are bound), grappa will raise a RuntimeError

impropers = [(2,3,4,5)]

try:
    methane = Molecule(atoms=atoms, bonds=bonds, impropers=impropers, partial_charges=partial_charges, atomic_numbers=atomic_numbers)
except RuntimeError as e:
    print(e)
# %%
# now we can obtain a dgl graph from the molecule that contains the idx tuples of the atoms that are involved in each term. This can be used as input for the grappa models:
g = methane.to_dgl()
print('atom ids:', g.nodes['n1'].data['ids'])
print('angle idxs: ', g.nodes['n3'].data['idxs'])
# %%
# we can save this as json or compressed npz file:
methane.to_json('methane.json')
# methane.save('methane.npz')

loaded_mol = Molecule.from_json('methane.json')
#%%

"""
CONSTRUCTION FROM OPENMM TOPOLOGY AND SYSTEM

The topology is used to obtain the atom ids, atomic numbers and the bonds while the system is used to obtain the partial charges and the impropers. Then these quantities are passed to the constructor described above.

Lets take the example of T4 lysozyme:
"""

from openmm.app import PDBFile, ForceField
import numpy as np
#%%
pdb = PDBFile('T4.pdb')
ff = ForceField('amber99sbildn.xml', 'tip3p.xml')

topology = pdb.topology
system = ff.createSystem(topology)
#%%
# now we can construct the molecule:
molecule = Molecule.from_openmm_system(openmm_system=system, openmm_topology=topology)
print('shape of impropers:', np.array(molecule.impropers).shape)
print('shape of propers:', np.array(molecule.propers).shape)
print('shape of angles:', np.array(molecule.angles).shape)
print('shape of bonds:', np.array(molecule.bonds).shape)
print('shape of atoms:', np.array(molecule.atoms).shape)
print('sum of partial charges:', np.sum(molecule.partial_charges))
# %%


# molecule and graph creation time is of the same order as the time it takes to create the openmm system using the classical force field:
import time
start = time.time()
pdb = PDBFile('T4.pdb')
ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
topology = pdb.topology
system = ff.createSystem(topology)
end = time.time()
print(f'time to create openmm system: {(end-start)*1e3:.1f} ms')
molecule = Molecule.from_openmm_system(openmm_system=system, openmm_topology=topology)
g = molecule.to_dgl()
end2 = time.time()
print(f'time to create molecule and graph: {(end2-end)*1e3:.1f} ms')
# %%
