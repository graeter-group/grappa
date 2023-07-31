#%%
import openmm as mm
from openmm import app
import numpy as np
import sys
from openmm.app import StateDataReporter

from pathlib import Path

#%%

pdb_folder = "data/pep1/A"

def generate_states(pdb_folder, n_states=2, temperature=300, forcefield=mm.app.ForceField('amber99sbildn.xml')):


    # Load the PDB file
    pdb = app.PDBFile(str(Path(pdb_folder)/Path('pep.pdb')))


    # Setup OpenMM system
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False)
    integrator = mm.LangevinIntegrator(500, 1.0, 0.001)
    simulation = app.Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)

    # report temperature and pot energy:
    simulation.reporters.append(StateDataReporter(sys.stdout, 5000, step=True, potentialEnergy=True, temperature=True))

    # store the temperature and pot energy in lists:
    temps = []
    pot_energies = []
    

    openmm_energies = []
    openmm_forces = []
    positions = []
    atomic_numbers = [atom.element.atomic_number for atom in pdb.topology.atoms()]

    # Sampling states with OpenMM and calculating energies and forces
    for _ in range(n_states):
        integrator.setTemperature(500)
        # 10000 steps of MD at 500K: get out of a local minimum
        simulation.step(50000)

        # 10000 steps of MD at the given temperature
        integrator.setTemperature(temperature)
        simulation.step(50000)

        state = simulation.context.getState(getEnergy=True, getForces=True)
        openmm_energies.append(state.getPotentialEnergy().value_in_unit(mm.unit.kilocalories_per_mole))
        openmm_forces.append(state.getForces(asNumpy=True).value_in_unit(mm.unit.kilocalories_per_mole/mm.unit.angstrom))

        # Save this configuration
        pos = simulation.context.getState(getPositions=True).getPositions().value_in_unit(mm.unit.angstrom)
        positions.append(pos)

generate_states(pdb_folder, n_states=3)
# %%
