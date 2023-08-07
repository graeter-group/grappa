#%%
# test a ff that resultet from copying the atom types of dop and hyp of the collagen ff, shifted them and inserted by hand into amber ff. for this, we have to insert at 3 places:
# 1. type definietion at the very top
# 2. residue definition
# 3. atom type nonbonded-params definition


import openmm
from openmm.app import ForceField as OpenMMFF
from grappa.PDBData.PDBMolecule import PDBMolecule
from pathlib import Path

#%%
mol = PDBMolecule.get_example()
pdb = mol.to_openmm()
# %%
ff1 = OpenMMFF("amber99sbildn.xml")

ff2 = OpenMMFF(str(Path(__file__).parent/"amber99sbildn_modified.xml"))
# %%
print("energies are close for normal peptides:")
data1 = mol.get_ff_data(ff1)
# %%
data2 = mol.get_ff_data(ff2)
# %%
import numpy as np

for k1, k2 in zip(data1, data2):
    print(np.allclose(k1, k2, atol=1e-5))
#%%
print("energies are off for the collagen ff:")
data_col = mol.get_ff_data(OpenMMFF("collagen_ff.xml"))
for k1, k2 in zip(data_col, data1):
    print(np.allclose(k1, k2, atol=1e-5))
print(data_col[0]- data_col[0].mean())
print(data1[0] - data1[0].mean())
#%%
# now with dop:
from openmm.app import PDBFile
mol_dop = PDBMolecule.from_pdb("data/ref_dipeptides/AJ/pep.pdb")
# %%
ff2.createSystem(pdb.topology)
#%%
try:
    ff1.createSystem(pdb.topology)
    raise
except ValueError:
    print("dop not in amber ff")    
# %%
data2 = mol_dop.get_ff_data(ff2)
# %%
data_col = mol_dop.get_ff_data(OpenMMFF("collagen_ff.xml"))
# %%
for k1, k2 in zip(data2, data_col):
    print(np.allclose(k1, k2, atol=1e-5))
print(np.abs(data2[1]).mean())
print(np.abs(data_col[1]).mean())
# %%
# collagen ff gives wrong result, rather trust the modified amber.
# %%
# now something with hyp:

# %%
mol_dop = PDBMolecule.from_pdb("data/ref_dipeptides/AO/pep.pdb")
# %%
ff2.createSystem(pdb.topology)
#%%
try:
    ff1.createSystem(pdb.topology)
    raise
except ValueError:
    print("hyp not in amber ff")    
# %%
data2 = mol_dop.get_ff_data(ff2)
# %%
data_col = mol_dop.get_ff_data(OpenMMFF("collagen_ff.xml"))
# %%
for k1, k2 in zip(data2, data_col):
    print(np.allclose(k1, k2, atol=1e-5))
print(np.abs(data2[1]).mean())
print(np.abs(data_col[1]).mean())
# %%
# collagen ff gives wrong result, rather trust the modified amber.
# %%
