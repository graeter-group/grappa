#%%
from pathlib import Path
import numpy as np
import torch

dspath = Path(__file__).parents[1]/'data'/"datasets"/"spice-des-monomers"

data = np.load(dspath/"111.npz")
smiles = data['mapped_smiles'].item()
charges = data['am1bcc_elf_charges']
# %%
from grappa.data import Molecule, MolData

mol1 = Molecule.from_smiles(smiles, openff_forcefield='openff_unconstrained-1.2.0.offxml', partial_charges=charges)
# %%
g1 = mol1.to_dgl()
# g1.nodes['n3'].data['idxs']
#%%
mol2 = Molecule.from_smiles(smiles, openff_forcefield='openff-1.2.0.offxml', partial_charges=charges)
g2 = mol2.to_dgl()
all_same = True
for ntype in g2.ntypes:
    for feature in g2.nodes[ntype].data.keys():
        if not torch.all(g1.nodes[ntype].data[feature] == g2.nodes[ntype].data[feature]):
            all_same = False
print(f"Unconstrained and constrained forcefield give the same graph structure:\n{all_same}")
# %%
data = {k:v for k,v in data.items()}
moldata = MolData.from_data_dict(data, partial_charge_key='am1bcc_elf_charges', forcefield='openff-1.2.0.offxml')
# %%
import matplotlib.pyplot as plt
plt.scatter(moldata.gradient.flatten(), moldata.gradient_ref.flatten())
g = moldata.to_dgl()
assert np.all(g.nodes['n3'].data['eq_ref'].numpy() == moldata.classical_parameters.angle_eq)
# %%
moldata2 = MolData.from_data_dict(data, partial_charge_key='am1bcc_elf_charges', forcefield='openff_unconstrained-1.2.0.offxml')
g2 = moldata2.to_dgl()
all_same = True
for ntype in g2.ntypes:
    for feature in g2.nodes[ntype].data.keys():
        if not torch.all(g.nodes[ntype].data[feature] == g2.nodes[ntype].data[feature]):
            all_same = False
print(f"Unconstrained and constrained forcefield give the same parameters:\n{all_same}")
# %%
print("The nonbonded parameters, however, are the same.")
# %%
