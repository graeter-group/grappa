#%%

# imports:
from grappa.ff import ForceField
from grappa.ff_utils.classical_ff.collagen_utility import get_collagen_forcefield
import openmm.unit
from grappa.constants import TopologyDict, ParamDict


# path to the grappa force field:
mpath = "/hits/fast/mbm/seutelf/grappa/mains/runs/stored_models/tutorial/best_model.pt"


# initialize the force field:
ff = ForceField(model_path=mpath)

#%%

# load example data:
from openmm.app import PDBFile
pdb = PDBFile("pep.pdb")
#%%
# get a system:
sys = ff.createSystem(pdb.topology)
# %%

# set up a simulation:
from openmm import LangevinIntegrator
from openmm.app import Simulation
from openmm import unit
# %%
integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.001*unit.picoseconds)
simulation = openmm.app.Simulation(pdb.topology, sys,integrator)
# %%
# set the positions:
simulation.context.setPositions(pdb.positions)
# minimize:
simulation.minimizeEnergy()
# %%
# run a short simulation:
simulation.step(1000)
# %%
# get the gradients:
import numpy as np
state = simulation.context.getState(getForces=True)
forces = state.getForces(asNumpy=True)
print(np.abs(np.array(forces)).mean())
# %%
