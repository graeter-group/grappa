#%%
import numpy as np
from pathlib import Path
#%%

dspath = Path('/hits/fast/mbm/seutelf/old_data/datasets/PDBDatasets/AA_opt_rad/charge_heavy_col_ff_amber99sbildn_filtered')

for molfile in dspath.iterdir():
    if molfile.is_dir():
        continue
    data = np.load(molfile)
    data = {k:v for k,v in data.items()}
    print(data.keys())
    break
# %%
xyz = data['n1 xyz']
gradient = data['n1 grad_qm']
energy = data['g u_qm']
pdb = data['pdb'].tolist()
pdb
#%%
xyz.shape
energy.shape
# %%
import tempfile
from openmm.app import PDBFile

pdbstring = ''.join(pdb)
# %%
with tempfile.TemporaryDirectory() as tmp:
    pdbpath = str(Path(tmp)/'pep.pdb')
    with open(pdbpath, "w") as pdb_file:
        pdb_file.write(pdbstring)
    openmm_pdb = PDBFile(pdbpath)

openmm_pdb.topology
# %%
str(data['sequence'])
# %%
