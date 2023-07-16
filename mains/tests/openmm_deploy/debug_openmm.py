#%%

from grappa.ff import ForceField
import openmm.app
import openmm.unit

ff = ForceField(model_path="/hits/fast/mbm/seutelf/grappa/mains/runs/reworked_test/versions/6/best_model.pt", classical_ff=openmm.app.ForceField("amber99sbildn.xml"))

ff.units["angle"] = openmm.unit.radian

# %%
pdb_path = "AG/pep.pdb"

# %%
from openmm.app import PDBFile
top = PDBFile(pdb_path).topology
from openmm.app import ForceField as openmm_ff

#%%

ff = ForceField(model=ForceField.get_classical_model(top, openmm.app.ForceField("amber99sbildn.xml")))
#%%
ff.units["angle"] = openmm.unit.degree
sys = ff.createSystem(topology=top, allow_radicals=False)

from openmm.app import Simulation
from openmm import unit
from openmm import LangevinIntegrator
from openmm.app import PDBReporter
# create a pdb reporter:

pdb_reporter = PDBReporter("test.pdb", 10)

# do a short simulation after an energy minimization:
integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.001*unit.picoseconds)
# sys.removeForce(4)
# sys.removeForce(3)
# sys.removeForce(1)
# sys.removeForce(0)
sys.getForces()

#%%
simulation = Simulation(top, sys, integrator)
simulation.context.setPositions(PDBFile(pdb_path).getPositions())
simulation.reporters.append(pdb_reporter)
simulation.step(100)
# simulation.minimizeEnergy()

# %%


#%%
from grappa import units
import numpy as np
pos = PDBFile(pdb_path).getPositions()
pos = simulation.context.getState(getPositions=True).getPositions()
positions = np.array([openmm.unit.Quantity(pos, unit=unit.nanometer).value_in_unit(units.DISTANCE_UNIT)])
# %%
delete_ft = ["NonbondedForce", "HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce"]
delete_ft = []
e_grappa, grad_grappa = ff.get_energies(top, positions, delete_force_type=delete_ft)
e_amber, grad_amber = ff.get_energies(top, positions, class_ff=openmm.app.ForceField("amber99sbildn.xml"), delete_force_type=delete_ft)
# %%
e_grappa, e_amber
# %%
diff = np.abs(grad_amber - grad_grappa)
diff, diff.var()
# %%
# do a scatter plot of the gradients:
import matplotlib.pyplot as plt
plt.scatter(grad_amber.flatten(), grad_grappa.flatten())

#%%
sys2 = openmm.app.ForceField("amber99sbildn.xml").createSystem(top)
# %%
for force in sys.getForces():
    if "Torsion" in force.__class__.__name__:
        for i in range(force.getNumTorsions()):
            print(force.getTorsionParameters(i))
            # if i == 5: break
        print()

for force in sys2.getForces():
    if "Torsion" in force.__class__.__name__:
        for i in range(force.getNumTorsions()):
            if i > 51:
                print(force.getTorsionParameters(i))
            # if i == 5: break
        print()

# %%
for force in sys.getForces():
    if "Torsion" in force.__class__.__name__:
        print(force.__class__.__name__, force.getNumTorsions())

for force in sys2.getForces():
    if "Torsion" in force.__class__.__name__:
        print(force.__class__.__name__, force.getNumTorsions())
# %%
# %%
sys1 = ff.createSystem(topology=top, allow_radicals=False)
# %%
sys2 = openmm.app.ForceField("amber99sbildn.xml").createSystem(top)

# %%
import grappa
t_ff = []
t_amber = []
for force in sys1.getForces():
    if isinstance(force, openmm.PeriodicTorsionForce):
        for i in range(force.getNumTorsions()):
            params = force.getTorsionParameters(i)
            if params[6].value_in_unit(grappa.units.OPENMM_TORSION_K_UNIT) != 0:
                print(params)
                t_ff.append(params)
#%%
for force in sys2.getForces():
    if isinstance(force, openmm.PeriodicTorsionForce):
        for i in range(force.getNumTorsions()):
            params = force.getTorsionParameters(i)
            if params[6].value_in_unit(grappa.units.OPENMM_TORSION_K_UNIT) != 0:
                print(params)
                t_amber.append(params)

# %%
t_ff = np.array(t_ff)
t_amber = np.array(t_amber)

# %%
t_ff.shape
#%%
t_amber.shape
# %%

# %%
