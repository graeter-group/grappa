#%%

# imports:
from grappa.ff import ForceField
import openmm.unit
import openmm.app



# initialize the force field from a tag:
ff = ForceField.from_tag("latest")

# ff = openmm.app.ForceField('amber99sbildn.xml', 'tip3p.xml') # uncomment for comparison
#%%

# load example data:
from openmm.app import PDBFile, Modeller

# load pdb and add hydrogens:
pdb = PDBFile("input_data/1aki.pdb")
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addHydrogens(openmm.app.ForceField('amber99sbildn.xml', 'tip3p.xml'))
# remove water:
modeller.deleteWater()
top = modeller.getTopology()
positions = modeller.getPositions()
#%%
# get a system:
sys = ff.createSystem(top)
# %%

# run a short simulation with these parameters:
# set up a simulation:
from openmm import LangevinIntegrator
from openmm.app import Simulation
from openmm import unit
# %%
# initialize an integrator with  temperature, heat-bath-coupling and timestep:
integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.001*unit.picoseconds)
simulation = Simulation(top, sys, integrator)
# %%
# set the positions:
simulation.context.setPositions(positions)
# minimize:
simulation.minimizeEnergy()
# %%
simulation.step(1000)
# keep track of the trajectory from now on:
simulation.reporters.append(openmm.app.PDBReporter('trajectory.pdb', 10))
from sys import stdout
simulation.reporters.append(openmm.app.StateDataReporter(stdout, 100, potentialEnergy=True, temperature=True))
simulation.step(1000)
# %%

import mdtraj as md
import nglview as nv

# Load the PDB file using MDTraj
traj = md.load('trajectory.pdb')

# Create a view for the trajectory
view = nv.show_mdtraj(traj)
# configure this to the stick model:
view.add_representation('ball+stick', selection='all')
# remove the cartoon model:
view.remove_cartoon()

# Display the view
view
# %%

# %%
