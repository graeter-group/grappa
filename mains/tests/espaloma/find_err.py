#%%
from grappa.PDBData.PDBMolecule import PDBMolecule

mol = PDBMolecule.load("monomer_with_esp.npz")
# %%
k_ref = mol.graph_data["n2"]["k_ref"]
k_esp = mol.graph_data["n2"]["k_esp"]
# %%
print(k_ref.mean())
print(k_esp.mean())
# %%
eq_ref = mol.graph_data["n2"]["eq_ref"]
eq_esp = mol.graph_data["n2"]["eq_esp"]

print(eq_ref.mean())
print(eq_esp.mean())
# %%
k_ref = mol.graph_data["n3"]["k_ref"]
k_esp = mol.graph_data["n3"]["k_esp"]

print(k_ref.mean())
print(k_esp.mean())
# %%
eq_ref = mol.graph_data["n3"]["eq_ref"]
eq_esp = mol.graph_data["n3"]["eq_esp"]

print(eq_ref.mean())
print(eq_esp.mean())
# %%
k_ref = mol.graph_data["n4"]["k_ref"]
k_esp = mol.graph_data["n4"]["k_esp"]

print(k_ref.max())
print(k_esp.max())
# %%
