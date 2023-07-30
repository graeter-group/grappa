def generate_states(pdb_folder, n_states=10):
    import openmm as mm
    from openmm import app
    import numpy as np
    from ase.io import read, write
    from ase import units as ase_units
    from openmm.app import StateDataReporter
    import sys
    from pathlib import Path

    #%%

    # Load the PDB file
    pdb = app.PDBFile(str(Path(pdb_folder)/Path('pep.pdb')))

    # Setup OpenMM system
    forcefield = app.ForceField('amber99sbildn.xml')
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False)
    integrator = mm.LangevinIntegrator(400, 1.0, 0.001)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    # simulation.reporters.append(StateDataReporter(sys.stdout, 500, step=True, potentialEnergy=True, temperature=True))

    simulation.step(10000)
    openmm_energies = []
    openmm_forces = []
    positions = []
    atomic_numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]

    # Sampling states with OpenMM and calculating energies and forces
    for _ in range(n_states):
        # 50000 steps of MD
        simulation.step(10000)

        state = simulation.context.getState(getEnergy=True, getForces=True)
        openmm_energies.append(state.getPotentialEnergy().value_in_unit(mm.unit.kilocalories_per_mole))
        openmm_forces.append(state.getForces(asNumpy=True).value_in_unit(mm.unit.kilocalories_per_mole/mm.unit.angstrom))

        # Save this configuration
        pos = simulation.context.getState(getPositions=True).getPositions().value_in_unit(mm.unit.angstrom)
        positions.append(pos)

    # store the states:
    np.save(str(Path(pdb_folder)/Path("atomic_numbers.npy")), atomic_numbers)
    np.save(str(Path(pdb_folder)/Path("positions.npy")), positions)
    np.save(str(Path(pdb_folder)/Path("openmm_energies.npy")), openmm_energies)
    np.save(str(Path(pdb_folder)/Path("openmm_forces.npy")), openmm_forces)


def generate_all_states(folder):
    from pathlib import Path
    for i, pdb_folder in enumerate(Path(folder).iterdir()):
        if pdb_folder.is_dir():
            print(f"generating states for {i}")
            try:
                generate_states(pdb_folder, n_states=10)
            except:
                print(f"failed to generate states for {i}")
                pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate states for a given folder.')
    parser.add_argument('--folder', type=str, help='The folder containing the PDB files.', default="pep3")
    args = parser.parse_args()
    generate_all_states(args.folder)