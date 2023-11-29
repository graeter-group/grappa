#%%
from grappa.data import Dataset, GraphDataLoader
import torch
from pathlib import Path
from grappa.utils import dgl_utils

# %%
dspath = Path(__file__).parents[2]/'data'/"dgl_datasets"

ds = Dataset.load(dspath/'rna-diverse')

# use openff 2.0.0 to predict forces and calculate their rmse wrt to the reference:

g, _ = ds[0]
print(g.nodes['n1'].data.keys())

gradients_openff = []
gradients_ref = []
actual_grad_openff = []

from grappa.utils import openmm_utils, openff_utils


for g, _ in ds:
    grad_ref = g.nodes['n1'].data['gradient_ref'].numpy()
    grad_openff = g.nodes['n1'].data['gradient_openff-1.2.0'].numpy()

    smiles = g.nodes['g'].data['mapped_smiles'].item()
    xyz = g.nodes['n1'].data['xzy'].numpy()

    sys, _,_ = openff_utils.get_openmm_system(mapped_smiles)

    en, forces = openmm_utils.get_energies(sys, xyz)

    actual_grad_openff = -forces

    sys = openmm_utils.remove_forces_from_system(keep='nonbonded')

    nonbonded_force, _ = openmm_utils.get_energies(sys, xyz)
    nonbonded_grad = -nonbonded_force

    gradients_ref.append(grad_ref)
    grad_openff.append(grad_openff - nonbonded_grad)
    grad_openff.append(actual_grad_openff - nonbonded_grad)
