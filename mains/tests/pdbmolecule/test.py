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
print(f"parametrizing takes {time.time() - start} seconds")

start = time.time()
g = mol.to_dgl()
print(f"to_dgl takes {time.time() - start} seconds")
# %%