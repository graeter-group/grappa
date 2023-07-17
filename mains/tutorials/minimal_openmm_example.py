#%%

# imports:
from grappa.ff import ForceField
import openmm.unit
import openmm.app



# initialize the force field from a tag:
ff = ForceField.from_tag("example")

# ff = openmm.app.ForceField('amber99sbildn.xml', 'tip3p.xml') # uncomment for comparison
#%%

# load example data:
from openmm.app import PDBFile
pdb = PDBFile("pep.pdb")
#%%
# get a system:
sys = ff.createSystem(pdb.topology)
# %%

# run a short simulation with these parameters:
# set up a simulation:
from openmm import LangevinIntegrator
from openmm.app import Simulation
from openmm import unit
# %%
# initialize an integrator with  temperature, heat-bath-coupling and timestep:
integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.001*unit.picoseconds)
simulation = Simulation(pdb.topology, sys, integrator)
# %%
# set the positions:
simulation.context.setPositions(pdb.positions)
# minimize:
simulation.minimizeEnergy()
# %%
simulation.step(10000)
# keep track of the trajectory from now on:
simulation.reporters.append(openmm.app.PDBReporter('trajectory.pdb', 10))
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
