#%%
from grappa.utils import get_repo_dir
from grappa import as_openmm
from openmm.app import PDBFile
from pathlib import Path

pdbpath = get_repo_dir()/'examples/dataset_creation/tripeptide_example_data/pdb_0.pdb'
pdbpath = str(pdbpath)


#%%

this_dir = Path(__file__).parent

ff = as_openmm('grappa-1.3', base_forcefield='/hits/fast/mbm/hartmaec/workdir/FF99SBILDNPX_OpenMM/grappa_1-3-amber99_ff99SB.xml', plot_dir=this_dir)
# %%
top = PDBFile(str(pdbpath)).topology
system = ff.createSystem(top)
# %%
