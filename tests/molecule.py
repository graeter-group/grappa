"""
Create a grappa molecule from a smiles string: Use a small molecule forcefield to identify improper torsions and check whether they are consistent with openmm. This script should run without error.
"""
#%%
from grappa.data import Molecule

# %%
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

#%%
# some smiles string:
smilestring = 'CC(C)(C)C(=O)OC1CC2C3CCC4=CC(=O)CCC4=C3C(CCC12C)C(=O)OC(C)(C)C'


from openff.toolkit import ForceField
from openff.toolkit import Molecule as OpenFFMolecule
from openff.units import unit

openff_mol = OpenFFMolecule.from_smiles(smilestring, allow_undefined_stereo=True)

# set charges to zero to make it faster.
openff_mol.partial_charges = [0] * len(openff_mol.atoms) * unit.elementary_charge

# %%
ff = ForceField('openff_unconstrained-1.2.0.offxml')
system = ff.create_openmm_system(openff_mol.to_topology(), charge_from_molecules=[openff_mol])
# %%
mol = Molecule.from_openmm_system(system, openmm_topology=openff_mol.to_topology().to_openmm())
# %%
f = get_force(system, 'HarmonicAngleForce')
angles_openmm = get_idx_tuples(f)
angles_grappa = mol.angles
#%%
angles_grappa = [(a0, a1, a2) if a0 < a2 else (a2, a1, a0) for a0, a1, a2 in angles_grappa] # canonical sorting
angles_openmm = [(a0, a1, a2) if a0 < a2 else (a2, a1, a0) for a0, a1, a2 in angles_openmm] # canonical sorting
assert set(angles_openmm) == set(angles_grappa), "Not all openmm angles are contained in grappa."
# %%
f = get_force(system, 'PeriodicTorsionForce')
torsions_openmm = get_idx_tuples(f)
torsions_openmm = [(t0, t1, t2, t3) if t0 < t3 else (t3, t2, t1, t0) for t0, t1, t2, t3 in torsions_openmm] # canonical sorting
torsions_grappa = mol.propers
impropers_grappa = mol.impropers

# canonical sorting:
torsions_grappa = [(t0, t1, t2, t3) if t0 < t3 else (t3, t2, t1, t0) for t0, t1, t2, t3 in torsions_grappa]
impropers_grappa = [(t0, t1, t2, t3) if t0 < t3 else (t3, t2, t1, t0) for t0, t1, t2, t3 in impropers_grappa]

# NOTE: canonical sorting for impropers is flawed!

assert all([t in torsions_grappa or t in impropers_grappa for t in torsions_openmm]), "Not all openmm torsions are either contained in grappa or are improper."
# %%
assert len(mol.impropers) > 0
# %%