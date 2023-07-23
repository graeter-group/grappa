#%%
from grappa.PDBData.PDBMolecule import PDBMolecule

mol = PDBMolecule.get_example()
# %%
# time these commands:
# g = mol.parametrize()
# g = mol.to_dgl()
import time

start = time.time()
g = mol.parametrize()
print(time.time() - start)

start = time.time()
g = mol.to_dgl()
print(time.time() - start)
# %%