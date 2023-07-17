#%%
from grappa.PDBData.PDBMolecule import PDBMolecule
mol = PDBMolecule.get_example()
print(mol.gradients.shape)
print(mol.xyz.shape)
print(mol.energies.shape)
# %%

# some data investigation utility
from grappa.ff import ForceField
ff = ForceField.from_tag("example")

fig, ax = mol.compare_with_ff(ff, ff_title="grappa")

# %%
import openmm.app 
fig, ax = mol.compare_with_ff(openmm.app.ForceField("amber99sbildn.xml"), ff_title="amber")
# %%

# print some lines from the pdb file:
print(*mol.pdb[15:25])
# %%
# create an openmm topology:
top = mol.to_openmm().topology
# %%
# convert to a dgl graph:
print(type(mol.to_dgl()))
# %%
# parametrize the dgl graph with a classical forcefield:
g = mol.parametrize(openmm.app.ForceField("amber99sbildn.xml"))
# %%
# nonbonded energy contribution calculated by the classical forcefield:
print(g.nodes["g"].data["u_nonbonded_ref"].shape)

# equilibrium angles:
print(g.nodes["n3"].data["eq_ref"].shape)

# %%
