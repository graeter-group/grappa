#%%
from grappa.data import MolData
import torch
from pathlib import Path
from grappa.utils import dgl_utils
from grappa.constants import BONDED_CONTRIBUTIONS

# %%
dspath = Path(__file__).parents[1]/'data'/"grappa_datasets"

mol = MolData.load(dspath/'spice-des-monomers'/'1.npz')
g = mol.to_dgl()
# %%
def model(graph):
    """
    Model that simply write the parameters with suffix from some graph in the given graph.
    """
    suffix = '_ref'
    for lvl, param in BONDED_CONTRIBUTIONS:
        graph.nodes[lvl].data[param] = g.nodes[lvl].data[param+suffix]
    return graph

# %%
from grappa.openmm_wrapper import openmm_Grappa

openmm_grappa = openmm_Grappa(model)

# create an openmm system, calculate the energies, then predict the parameters with openmm_grappa and our simple model, and calculate the energies again. The energies should be the same:

#%%
from grappa.utils import openff_utils, openmm_utils
print(mol.mapped_smiles)
#%%
system,_,_ = openff_utils.get_openmm_system(mapped_smiles=mol.mapped_smiles, partial_charges=mol.molecule.partial_charges, openff_forcefield='openff_unconstrained-1.2.0.offxml')

en, grads = openmm_utils.get_energies(system, mol.xyz)

#%%