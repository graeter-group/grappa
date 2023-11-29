import numpy as np
from typing import Union, Dict
from pathlib import Path
import tempfile


def get_energies(openmm_system, xyz):
    """
    Returns enegries, forces. in units kcal/mol and kcal/mol/angstroem
    Assume that xyz is in angstroem and has shape (num_confs, num_atoms, 3).
    """
    from openmm import app
    import openmm
    from openmm import unit

    assert len(xyz.shape) == 3
    assert xyz.shape[1] == openmm_system.getNumParticles()
    assert xyz.shape[2] == 3

    # create a context:
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(openmm_system, integrator)

    energies = []
    forces = []
    # set positions:
    for pos in xyz:
        context.setPositions(unit.Quantity(pos, unit.angstrom))
        state = context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        forces_ = state.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole/unit.angstrom)
        energies.append(energy)
        forces.append(forces_)

    return np.array(energies), np.array(forces)


def remove_forces_from_system(system, remove=None, keep=None)->'openmm.System':
    """
    Modifies the OpenMM system by removing forces according to the 'remove' and 'keep' lists.
    Forces are identified by their class name.

    Parameters:
    - system: The OpenMM System object to modify.
    - remove: A list of strings. Forces with class names containing any of these strings will be removed.
    - keep: A list of strings. If not None, only forces with class names containing these strings will be kept.

    Returns:
    - system: The modified OpenMM System object.
    """

    # First, identify the indices of the forces to remove
    forces_to_remove = []
    for i, force in enumerate(system.getForces()):
        force_name = force.__class__.__name__.lower()
        if keep is not None:
            if not any(k.lower() in force_name for k in keep):
                forces_to_remove.append(i)
        elif remove is not None:
            if any(k.lower() in force_name for k in remove):
                forces_to_remove.append(i)

    # Remove the forces by index, in reverse order to not mess up the indices
    for i in reversed(forces_to_remove):
        system.removeForce(i)

    return system


def set_partial_charges(system, partial_charges:Union[list, np.ndarray])->'openmm.System':
    """
    Set partial charges of a system.
    """
    import openmm
    from openmm import unit

    # get the nonbonded force (behaves like a reference not a copy!):
    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            if not nonbonded_force is None:
                raise ValueError("More than one nonbonded force found.")
            nonbonded_force = force
            
    if nonbonded_force is None:
        raise ValueError("No nonbonded force found.")
    
    # set the charges:
    if len(partial_charges) != nonbonded_force.getNumParticles():
        raise ValueError("Number of partial charges does not match number of particles.")
    
    for i, charge in enumerate(partial_charges):
        # get the parameters:
        _, sigma, epsilon = nonbonded_force.getParticleParameters(i)
        # set the charge:
        nonbonded_force.setParticleParameters(i, charge=charge, sigma=sigma, epsilon=epsilon)

    return system


def write_to_system(system, parameters:'grappa.data.Parameters'):
    """
    Writes bonded parameters in an openmm system. For interactions that are already present in the system, overwrite the parameters; otherwise add the interaction to the system. The forces, however, must be already present in the system.
    The ids of the atoms, bonds, etc in the parameters object must be the same as the system indices.
    """
    import openmm
    from openmm.unit import Quantity
    from grappa import units as grappa_units

    # MAKE DEEPCOPIES OF THESE ALSO OF PARAMETERS.BONDS and angles etc
    # THEN REMOVE ADDED BONDS etc after finding them

    bonds = parameters.bonds
    angles = parameters.angles
    impropers = parameters.impropers
    propers = parameters.propers

    bond_ks = Quantity(parameters.bond_k, unit=grappa_units.BOND_K_UNIT)
    bond_eqs = Quantity(parameters.bond_eq, unit=grappa_units.BOND_EQ_UNIT)
    angle_ks = Quantity(parameters.angle_k, unit=grappa_units.ANGLE_K_UNIT)
    angle_eqs = Quantity(parameters.angle_eq, unit=grappa_units.ANGLE_EQ_UNIT)
    improper_ks = Quantity(parameters.improper_ks, unit=grappa_units.TORSION_K_UNIT)
    improper_phases = Quantity(parameters.improper_phases, unit=grappa_units.TORSION_PHASE_UNIT)
    proper_ks = Quantity(parameters.proper_ks, unit=grappa_units.TORSION_K_UNIT)
    proper_phases = Quantity(parameters.proper_phases, unit=grappa_units.TORSION_PHASE_UNIT)

    assert np.all(proper_ks >= 0)
    assert np.all(improper_ks >= 0)

    # create a dictionary because we will need lookups and dict lookup is more efficient than list.index (these are shallow copies)
    bond_lookup = {tuple(b):(bond_ks[i], bond_eqs[i]) for i, b in enumerate(bonds)}
    angle_lookup = {tuple(a): (angle_ks[i], angle_eqs[i]) for i, a in enumerate(angles)}
    improper_lookup = {tuple(i) for i, i in enumerate(impropers)}
    proper_lookup = {tuple(p) for i, p in enumerate(propers)}

    ordered_torsions = {sorted(tuple(p)) for p in list(improper_lookup) + list(proper_lookup)}

    # loop through the system forces, overwrite the parameters if present, otherwise add the interaction.
    # in each step, first transform the parameter id to the system index.

    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                # Get the atom indices and existing parameters
                atom1, atom2, length, k = force.getBondParameters(i)

                # try both orderings:
                bond_param = bond_lookup.get((atom1, atom2), None)
                if bond_param is None:
                    bond_param = bond_lookup.get((atom2, atom1), None)
                    if not bond_param is None:
                        bond_lookup.pop((atom2, atom1))
                else:
                    bond_lookup.pop((atom1, atom2))

                if not bond_param is None:
                    # Update the parameters
                    new_length, new_k = bond_param
                    force.setBondParameters(i, atom1, atom2, new_length, new_k)
            force.updateParametersInContext(system.context)

        elif isinstance(force, openmm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                atom1, atom2, atom3, _, _ = force.getAngleParameters(i)

                angle_param = angle_lookup.pop((atom1, atom2, atom3), None)
                if angle_param is None:
                    angle_param = angle_lookup.pop((atom3, atom2, atom1), None)

                if not angle_param is None:
                    force.setAngleParameters(i, atom1, atom2, atom3, angle_param[1], angle_param[0])
            force.updateParametersInContext(system.context)


        # check whether torsion is contained in both proper or improper. if so, set its k to zero, effectively removing the force.
        if isinstance(force, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                atom1, atom2, atom3, atom4, periodicity, phase, k = force.getTorsionParameters(i)

                # Check in proper and improper torsions
                if sorted((atom1, atom2, atom3, atom4)) in ordered_torsions:
                    # Set k to zero or update parameters
                    force.setTorsionParameters(i, atom1, atom2, atom3, atom4, periodicity, phase, 0)
            force.updateParametersInContext(system.context)

    
        # now add the bonds and angles that have not been added yet as new forces.
        # also add a new torsion force, one for proper, one for improper.

    # Adding remaining bonds
    if bond_lookup:
        new_bond_force = openmm.HarmonicBondForce()
        for bond, params in bond_lookup.items():
            new_bond_force.addBond(bond[0], bond[1], length=params[1], k=params[0])
        system.addForce(new_bond_force)

    # Adding remaining angles
    if angle_lookup:
        new_angle_force = openmm.HarmonicAngleForce()
        for angle, params in angle_lookup.items():
            new_angle_force.addAngle(angle[0], angle[1], angle[2], angle=params[1], k=params[0])
        system.addForce(new_angle_force)

    # Adding all torsions:
    proper_torsion_force = openmm.PeriodicTorsionForce()
    for i, torsion in enumerate(impropers):
        for n in range(1, len(improper_ks[i])):
            proper_torsion_force.addTorsion(torsion[0], torsion[1], torsion[2], torsion[3], periodicity=n, phase=improper_phases[i][n], k=improper_ks[i][n])

    # Adding all impropers:
    improper_torsion_force = openmm.PeriodicTorsionForce()
    for i, torsion in enumerate(impropers):
        for n in range(1, len(improper_ks[i])):
            improper_torsion_force.addTorsion(torsion[0], torsion[1], torsion[2], torsion[3], periodicity=n, phase=improper_phases[i][n], k=improper_ks[i][n])

    system.addForce(proper_torsion_force)
    system.addForce(improper_torsion_force)

    return system



def topology_from_pdb(pdbstring:str)->'openmm.Topology':
    """
    Returns an openmm topology from a pdb string in which the lines are separated by '\n'.
    """
    from openmm.app import PDBFile

    with tempfile.TemporaryDirectory() as tmp:
        pdbpath = str(Path(tmp)/'pep.pdb')
        with open(pdbpath, "w") as pdb_file:
            pdb_file.write(pdbstring)
        openmm_pdb = PDBFile(pdbpath)

    return openmm_pdb.topology