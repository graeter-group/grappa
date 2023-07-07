#%%
from openmm.app import PDBFile, Element
from openmm import unit

SHIFT = 0

def pdb_to_lists(file_path, shift_idx=0):
    # Load the pdb file using OpenMM's PDBFile
    pdb = PDBFile(file_path)
    
    # Get the topology
    topology = pdb.getTopology()

    # Prepare the atoms and bonds lists
    atoms = []
    bonds = []

    # Iterate over the atoms in the topology
    for atom in topology.atoms():
        # Add the atom data to the atoms list
        atoms.append([atom.index+shift_idx, atom.name, atom.residue.name, atom.residue.index, [0,0], atom.element.atomic_number])

    # Iterate over the bonds in the topology
    for bond in topology.bonds():
        # Add the bond data to the bonds list
        bonds.append([bond[0].index+shift_idx, bond[1].index+shift_idx])

    return atoms, bonds

atoms, bonds = pdb_to_lists("AG/pep.pdb", shift_idx=SHIFT)

# %%
from grappa.ff import ForceField
from openmm.app import ForceField as openmm_ff
import openmm.unit
from openmm.app import PDBFile

ref_ff = openmm_ff("amber99sbildn.xml")
#ff = ForceField(model_path="/hits/fast/mbm/seutelf/ml_ff/mains/runs/param_search/versions/4_dotted_700_6_6/best_model.pt", classical_ff=ref_ff)

pdb_path = "AG/pep.pdb"
top = PDBFile(pdb_path).topology

model = ForceField.get_classical_model(top, ref_ff) # this model imitates the amber99sbildn forcefield. it is for testing only.

ff = ForceField(model=model, classical_ff=openmm.app.ForceField("amber99sbildn.xml"))

params = ff({"atoms":atoms, "bonds":bonds})
# %%
print("dict keys and mean values:")
print(*[(k, v.mean()) for (k,v) in zip(params.keys(), params.values())], sep="\n")
# %%
print(params["bond_idxs"])
# %%
params2 = ff(pdb_path)
# %%
import numpy as np
print("difference of parameters if done by topology:")
for (k,v) in zip(params.values(), params2.values()):
    print(np.abs((k-v)).mean())

# %%

pdb_path = "AG_rad/pep.pdb"
top = PDBFile(pdb_path).topology

atoms, bonds = pdb_to_lists("AG_rad/pep.pdb", shift_idx=SHIFT)

model = ForceField.get_classical_model(top, ref_ff) # this model imitates the amber99sbildn forcefield. it is for testing only.

ff = ForceField(model=model, classical_ff=openmm.app.ForceField("amber99sbildn.xml"))

params = ff({"atoms":atoms, "bonds":bonds, "radicals":[17]})
params2 = ff(pdb_path)
# %%
print("difference of parameters for radical if done by topology")
for (k,v) in zip(params.values(), params2.values()):
    print(np.abs((k-v)).mean())

# %%
ff = ForceField(model=ForceField.get_zero_model())
# %%
params = ff({"atoms":atoms, "bonds":bonds, "radicals":[17]})
# %%
print("dict keys and mean values:")
print(*[(k, v.mean()) for (k,v) in zip(params.keys(), params.values())], sep="\n")
# %%
params["bond_k"]
# %%
