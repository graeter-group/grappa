#%%
from grappa.ff import ForceField
import openmm.app
import openmm.unit
from openmm import unit
from grappa import units
import numpy as np


# %%
pdb_path = "AG/pep.pdb"
from openmm.app import PDBFile
top = PDBFile(pdb_path).topology
pos = PDBFile(pdb_path).getPositions()
positions = np.array([openmm.unit.Quantity(pos, unit=unit.nanometer).value_in_unit(units.DISTANCE_UNIT)])


ff = ForceField(model=ForceField.get_classical_model(top, openmm.app.ForceField("amber99sbildn.xml")))
# %%
delete_ft = []
e_grappa, grad_grappa = ff.get_energies(top, positions, delete_force_type=delete_ft)
e_amber, grad_amber = ff.get_energies(top, positions, class_ff=openmm.app.ForceField("amber99sbildn.xml"), delete_force_type=delete_ft)
print(e_amber, e_grappa)
# %%
import matplotlib.pyplot as plt
plt.scatter(grad_amber.flatten(), grad_grappa.flatten())
plt.xlabel("amber")
plt.ylabel("grappa")
#%%
from grappa.PDBData.PDBMolecule import PDBMolecule
from grappa.models import energy
mol = PDBMolecule.from_pdb(pdb_path)
g = mol.parametrize()
#%%
def compare_energies(g, ff, slight_change=False):
    e_amber = g.nodes["g"].data["u_total_ref"] - g.nodes["g"].data["u_nonbonded_ref"]
    e_amber
    import torch
    with torch.no_grad():
        g = ff.model(g)
        if slight_change:
            g.nodes["n3"].data["k"] *= 1.05
        e_writer = energy.WriteEnergy()
        g = e_writer(g)
    e_grappa = g.nodes["g"].data["u"]
    return e_amber, e_grappa

e_amber, e_grappa = compare_energies(g, ff)

print("amber, grappa:\n")
print(e_amber.item(), e_grappa.item())
e_amber, e_grappa = compare_energies(g, ff, slight_change=True)
print("with slight change of grappa params:")
print(e_amber.item(), e_grappa.item())
#%%