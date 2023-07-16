#%%
from grappa.ff import ForceField
import openmm.app
import openmm.unit
from openmm import unit
from grappa import units
import numpy as np

model_path = "/hits/fast/mbm/seutelf/grappa/mains/runs/reworked_test/versions/17_good/best_model.pt"

ff = ForceField(model_path=model_path, classical_ff=openmm.app.ForceField("amber99sbildn.xml"))



# %%
pdb_path = "AG/pep.pdb"
from openmm.app import PDBFile
top = PDBFile(pdb_path).topology
pos = PDBFile(pdb_path).getPositions()
#%% do a small simulation at 300K:
from openmm.app import Simulation
from openmm import unit
from openmm import LangevinIntegrator
integrator = LangevinIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.001*unit.picoseconds)
sys = openmm.app.ForceField("amber99sbildn.xml").createSystem(topology=top)
simulation = Simulation(top, sys, integrator)
simulation.context.setPositions(pos)
positions = []
for i in range(50):
    simulation.step(100)
    positions.append(simulation.context.getState(getPositions=True).getPositions())



positions = np.array([openmm.unit.Quantity(pos, unit=unit.nanometer).value_in_unit(units.DISTANCE_UNIT) for pos in positions])


# %%
delete_ft = ["angle"]
delete_ft = []
e_grappa, grad_grappa = ff.get_energies(top, positions, delete_force_type=delete_ft)
e_amber, grad_amber = ff.get_energies(top, positions, class_ff=openmm.app.ForceField("amber99sbildn.xml"), delete_force_type=delete_ft)
print("means:", e_amber.mean(), e_grappa.mean())
print("stds:", e_amber.std(), e_grappa.std())
e_amber -= e_amber.mean()
e_grappa -= e_grappa.mean()
diff = np.abs(e_grappa - e_amber)
print("max diff:", diff.max())
print("mean diff:", diff.mean())
# %%
import matplotlib.pyplot as plt
plt.scatter(grad_amber.flatten(), grad_grappa.flatten())
plt.plot(grad_grappa.flatten(), grad_grappa.flatten(), color="black")
plt.xlabel("amber")
plt.ylabel("grappa")


#%%
plt.scatter(e_amber, e_grappa)
plt.plot(e_grappa, e_grappa, color="black")
#%%
from grappa.PDBData.PDBMolecule import PDBMolecule
from grappa.models import energy

from grappa.run.run_utils import load_yaml
from pathlib import Path
d = load_yaml(Path(model_path).parent / "run_config.yml")

ds_path = d["ds_path"][0]
from grappa.PDBData.PDBDataset import PDBDataset
ds = PDBDataset.load_npz(Path(ds_path).parent / Path("base"))
# %%
mol = ds[0]
mol.filter_confs()
# %%
data_amber = mol.get_ff_data(openmm.app.ForceField("amber99sbildn.xml"))
data_grappa = mol.get_ff_data(ff)
data_qm = mol.energies, mol.gradients
# %%
e, f = data_grappa

e_amber, f_amber = data_amber

e_qm, f_qm = data_qm

e -= e.mean()
e_amber -= e_amber.mean()
e_qm -= e_qm.mean()
# %%

plt.scatter(f.flatten(), f_amber.flatten(), s=0.3)
plt.plot(f_amber.flatten(), f_amber.flatten(), color="black")
plt.xlim(-50,50)
plt.ylim(-50,50)
print("mae: ", np.abs(f.flatten() - f_amber.flatten()).mean())
print("rmse: ", np.sqrt(np.square(f.flatten() - f_amber.flatten()).mean()))

# %%
plt.scatter(e, e_amber)
plt.plot(e_amber, e_amber, color="black")
print("mae: ", np.abs(e - e_amber).mean())
print("rmse: ", np.sqrt(np.square(e - e_amber).mean()))
# %%

# comparison between QM and Grappa
plt.scatter(f.flatten(), f_qm.flatten(), s=0.3)
plt.plot(f_qm.flatten(), f_qm.flatten(), color="black")
plt.xlim(-50,50)
plt.ylim(-50,50)
print("mae: ", np.abs(f.flatten() - f_qm.flatten()).mean())
print("rmse: ", np.sqrt(np.square(f.flatten() - f_qm.flatten()).mean()))
# %%
plt.scatter(e, e_qm)
plt.plot(e_qm, e_qm, color="black")
print("mae: ", np.abs(e - e_qm).mean())
print("rmse: ", np.sqrt(np.square(e - e_qm).mean()))
# %%

# comparison between QM and Amber
plt.scatter(f_amber.flatten(), f_qm.flatten(), s=0.3)
plt.plot(f_qm.flatten(), f_qm.flatten(), color="black")
plt.xlim(-50,50)
plt.ylim(-50,50)
print("mae: ", np.abs(f_amber.flatten() - f_qm.flatten()).mean())
print("rmse: ", np.sqrt(np.square(f_amber.flatten() - f_qm.flatten()).mean()))
# %%
plt.scatter(e_amber, e_qm)
plt.plot(e_qm, e_qm, color="black")
print("mae: ", np.abs(e_amber - e_qm).mean())
print("rmse: ", np.sqrt(np.square(e_amber - e_qm).mean()))

# %%
