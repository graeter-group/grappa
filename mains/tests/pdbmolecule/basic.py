#%%

from pathlib import Path
import tempfile
import dgl
import numpy as np

from grappa.PDBData.PDBMolecule import PDBMolecule


# %%
pdb = Path(__file__).parent/"AG/pep.pdb"
mol = PDBMolecule.from_pdb(pdb)
# %%
g = mol.to_dgl()
# %%

g = mol.parametrize()
# %%
u_amber = g.nodes["g"].data["u_total_ref"].detach().numpy()
# %%
u_amber
# %%
grad_amber = g.nodes["n1"].data["grad_total_ref"].detach().numpy()
grad_amber.shape
# %%
