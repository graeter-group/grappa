import openmm as mm
from openmm import app
import numpy as np
from utils import ProgressReporter

from pathlib import Path

def generate_states(pdb_folder, n_states=10, temperature=300, forcefield=mm.app.ForceField('amber99sbildn.xml'), plot=False, between_steps=50000):

    # Load the PDB file
    pdb = app.PDBFile(str(Path(pdb_folder)/Path('pep.pdb')))

    # Setup OpenMM system
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False)
    integrator = mm.LangevinIntegrator(500, 1.0, 0.001)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    total_steps = n_states * between_steps + n_states * (between_steps//5)
    if plot:
        total_steps += 100
        
    simulation.reporters.append(ProgressReporter(1000, total_steps))

    if plot:
        simulation.step(100) # equilibrate a bit to reach the temperature
        from utils import CustomReporter
        simulation.reporters.append(CustomReporter(100))
        sampling_steps = []


    step = 0
    openmm_energies = []
    openmm_forces = []
    positions = []
    atomic_numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]

    # Sampling states with OpenMM and calculating energies and forces
    for _ in range(n_states):
        
        # between steps of MD at 500K: get out of a local minimum
        integrator.setTemperature(500)
        simulation.step(between_steps)
        step += between_steps

        # between_steps/10 steps of MD to reach the given temperature
        integrator.setTemperature(temperature)
        simulation.step(between_steps//5)
        step += between_steps//5

        state = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)

        # Save this configuration
        openmm_energies.append(state.getPotentialEnergy().value_in_unit(mm.unit.kilocalories_per_mole))
        openmm_forces.append(state.getForces(asNumpy=True).value_in_unit(mm.unit.kilocalories_per_mole/mm.unit.angstrom))

        pos = state.getPositions().value_in_unit(mm.unit.angstrom)
        positions.append(pos)

        if plot:
            sampling_steps.append(step)



    # store the states:
    np.save(str(Path(pdb_folder)/Path("atomic_numbers.npy")), atomic_numbers)
    np.save(str(Path(pdb_folder)/Path("positions.npy")), positions)
    np.save(str(Path(pdb_folder)/Path("openmm_energies.npy")), openmm_energies)
    np.save(str(Path(pdb_folder)/Path("openmm_forces.npy")), openmm_forces)

    # obtain the total charge:
    ############################
    nonbonded_force = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)]
    assert len(nonbonded_force) == 1
    nonbonded_force = nonbonded_force[0]

    total_charge = sum([nonbonded_force.getParticleParameters(i)[0].value_in_unit(mm.unit.elementary_charge) for i in range(nonbonded_force.getNumParticles())])


    if not np.isclose(total_charge, round(total_charge,0), atol=1e-5):
        raise ValueError(f"Total charge is not an integer: {total_charge}")

    total_charge = int(round(total_charge,0))
    ############################

    np.save(str(Path(pdb_folder)/Path("charge.npy")), np.array([total_charge]))

    if plot:
        for rep in simulation.reporters:
            if isinstance(rep, CustomReporter):
                rep.plot(sampling_steps=sampling_steps, filename=str(Path(pdb_folder)/Path("sampling.png")))
                break



def generate_all_states(folder, n_states=10, temperature=300, plot=False, between_steps=25000):
    from pathlib import Path
    for i, pdb_folder in enumerate(Path(folder).iterdir()):
        if pdb_folder.is_dir():
            print(f"generating states for {i}")
            try:
                generate_states(pdb_folder, n_states=n_states, temperature=temperature, plot=plot, between_steps=between_steps)
            except Exception as e:
                print("-----------------------------------")
                print(f"failed to generate states for {i} in {pdb_folder.stem}:{type(e)}:\n{e}")
                print("-----------------------------------\n")
                raise
                continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate states for a given folder.')
    parser.add_argument('folder', type=str, help='The folder containing the PDB files.', default="data/pep1")
    parser.add_argument('--n_states', '-n', type=int, help='The number of states to generate.', default=10)
    parser.add_argument('--temperature', '-t', type=int, help='The temperature to use for the simulation.', default=300)
    parser.add_argument('--plot', '-p', action='store_true', help='Whether to plot the sampling temperatures and potential energies.')
    parser.add_argument('--between_steps', '-b', type=int, help='The number of steps to take between the sampling steps.', default=50000)
    args = parser.parse_args()
    generate_all_states(args.folder, n_states=args.n_states, temperature=args.temperature, plot=args.plot, between_steps=args.between_steps)