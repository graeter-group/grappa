"""
Create a molecule and for bonds, angles and proper torsions: check whether our functions return the same up to equivalence by invariant permutations.
This script should urn without throwing an error.
"""
#%%
from grappa.utils import tuple_indices
from typing import Union, List
import numpy as np

#%%
import openmm

def get_force(system:openmm.System, force_name:str):
    force_ = None
    for force in system.getForces():
        if force_name.lower() in force.__class__.__name__.lower():
            assert force_ is None, "More than one force of the same type found in system."
            force_ = force
    assert force_ is not None, f"Force {force_name} not found in system, forces are:\n{[f.__class__.__name__ for f in system.getForces()]}"
    return force_

def get_idx_tuples(force:openmm.Force):
    idxs = []
    if force.__class__.__name__ == 'HarmonicBondForce':
        for i in range(force.getNumBonds()):
            idxs.append((force.getBondParameters(i)[0], force.getBondParameters(i)[1]))
    elif force.__class__.__name__ == 'HarmonicAngleForce':
        for i in range(force.getNumAngles()):
            idxs.append((force.getAngleParameters(i)[0], force.getAngleParameters(i)[1], force.getAngleParameters(i)[2]))
    elif force.__class__.__name__ == 'PeriodicTorsionForce':
        for i in range(force.getNumTorsions()):
            idxs.append((force.getTorsionParameters(i)[0], force.getTorsionParameters(i)[1], force.getTorsionParameters(i)[2], force.getTorsionParameters(i)[3]))
    elif force.__class__.__name__ == 'NonbondedForce':
        for i in range(force.getNumParticles()):
            idxs.append(i)
    else:
        raise NotImplementedError(f"Force {force.__class__.__name__} not implemented.")
    return idxs


# %%

smilestring = 'CC(C)(C)C(=O)OC1CC2C3CCC4=CC(=O)CCC4=C3C(CCC12C)C(=O)OC(C)(C)C'


# now obtain the indices of bonds, angles and torsions with openmm/openff:
import pkgutil

assert pkgutil.find_loader("openff.toolkit") is not None, "openff.toolkit must be installed to run this test."

from openff.toolkit import ForceField, Topology, Molecule
from openff.units import unit

openff_mol = Molecule.from_smiles(smilestring, allow_undefined_stereo=True)

# set charges to zero to make it faster.

openff_mol.partial_charges = [0] * len(openff_mol.atoms) * unit.elementary_charge

# %%
ff = ForceField('openff_unconstrained-1.2.0.offxml')
system = ff.create_openmm_system(openff_mol.to_topology(), charge_from_molecules=[openff_mol])

#%%
f = get_force(system, 'HarmonicBondForce')
bonds_openmm = get_idx_tuples(f)
grappa_tuples = tuple_indices.get_idx_tuples(bonds_openmm)
# %%
f = get_force(system, 'HarmonicAngleForce')
angles_openmm = get_idx_tuples(f)
angles_openmm = [(a0, a1, a2) if a0 < a2 else (a2, a1, a0) for a0, a1, a2 in angles_openmm] # canonical sorting
assert set(angles_openmm) == set([tuple(a) for a in grappa_tuples['angles']]), "Angles not equal."
# %%
f = get_force(system, 'PeriodicTorsionForce')
torsions_openmm = get_idx_tuples(f)
torsions_openmm = [(t0, t1, t2, t3) if t0 < t3 else (t3, t2, t1, t0) for t0, t1, t2, t3 in torsions_openmm] # canonical sorting
# assert all([t in grappa_tuples['propers'] for t in torsions_openmm]), "Not all openmm torsions are contained in grappa." # no. must not be equal, there could be impropers in openmm or propers with zero force constant that are not in openmm

d = tuple_indices.get_neighbor_dict(bonds_openmm)

assert all([t in grappa_tuples['propers'] or tuple_indices.is_improper(ids=t, central_atom_position=None, neighbor_dict=d) for t in torsions_openmm]), "Not all openmm torsions are either contained in grappa or are improper."
# %%



# now create a molecule with improper torsions and check whether they are correctly identified:
smilestring = 'C[C@H](C(=O)O)N'
openff_mol = Molecule.from_smiles(smilestring, allow_undefined_stereo=True)
openff_mol.partial_charges = [0] * len(openff_mol.atoms) * unit.elementary_charge
ff = ForceField('openff_unconstrained-1.2.0.offxml')
system = ff.create_openmm_system(openff_mol.to_topology(), charge_from_molecules=[openff_mol])
#%%
f = get_force(system, 'HarmonicBondForce')
bonds_openmm = get_idx_tuples(f)
#%%
f = get_force(system, 'PeriodicTorsionForce')
torsions_openmm = get_idx_tuples(f)
torsions_openmm = [(t0, t1, t2, t3) if t0 < t3 else (t3, t2, t1, t0) for t0, t1, t2, t3 in torsions_openmm] # canonical sorting
torsions_grappa = tuple_indices.get_idx_tuples(bonds_openmm)['propers']

d = tuple_indices.get_neighbor_dict(bonds_openmm)
# %%
assert len(torsions_openmm) > len(torsions_grappa), f"There are no improper torsions in the openmm system."
# %%
if not all([t in torsions_grappa or tuple_indices.is_improper(ids=t, central_atom_position=None, neighbor_dict=d) for t in torsions_openmm]):
    raise AssertionError("Not all openmm torsions are either contained in grappa or are improper.")
# %%
