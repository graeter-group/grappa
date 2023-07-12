#%%
pdb_path = "AG/pep.pdb"

# %%
from openmm.app import PDBFile
top = PDBFile(pdb_path).topology
from openmm.app import ForceField as openmm_ff

#%%
from grappa.ff import ForceField
ff = ForceField(model=ForceField.get_classical_model(top, openmm_ff("amber99sbildn.xml")))
#%%



#%%
sys2 = openmm_ff("amber99sbildn.xml").createSystem(top)
# %%
# for force in sys.getForces():
#     if "Torsion" in force.__class__.__name__:
#         for i in range(force.getNumTorsions()):
#             print(force.getTorsionParameters(i))
#             # if i == 5: break
#         print()

for force in sys2.getForces():
    if "Torsion" in force.__class__.__name__:
        for i in range(force.getNumTorsions()):
            if i > 51:
                print(force.getTorsionParameters(i))
            # if i == 5: break
        print()


for force in sys2.getForces():
    if "Torsion" in force.__class__.__name__:
        print(force.__class__.__name__, force.getNumTorsions())
# %%