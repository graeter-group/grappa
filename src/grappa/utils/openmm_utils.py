

def get_energies(openmm_system, xyz):
    """
    Returns enegries, forces.
    Assume that xyz is in angstroem.
    """
    from openmm import app
    import numpy as np
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


def remove_forces_from_system(system, exclude=None, keep=None):
    """
    Modifies the OpenMM system by removing forces according to the 'exclude' and 'keep' lists.
    Forces are identified by their class name.

    Parameters:
    - system: The OpenMM System object to modify.
    - exclude: A list of strings. Forces with class names containing any of these strings will be removed.
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
        elif exclude is not None:
            if any(k.lower() in force_name for k in exclude):
                forces_to_remove.append(i)

    # Remove the forces by index, in reverse order to not mess up the indices
    for i in reversed(forces_to_remove):
        system.removeForce(i)

    return system
