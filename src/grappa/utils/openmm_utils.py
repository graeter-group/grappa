import numpy as np
from typing import Union


def get_energies(openmm_system, xyz):
    """
    Returns enegries, forces.
    Assume that xyz is in angstroem.
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


def remove_forces_from_system(system, remove=None, keep=None):
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
        nonbonded_force.setParticleParameters(i, charge=unit.Quantity(charge, unit.elementary_charge))

    return system