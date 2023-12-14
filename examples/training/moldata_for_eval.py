"""
How to create a MolData object just for quick evaluation, without training or ensuring that the test data is unseen.
"""
#%%
from grappa.data import MolData, Molecule
import numpy as np
#%%
# create a molecule object
atoms = [1, 2, 3, 4, 5]
bonds = [(1, 2), (1, 3), (1, 4), (1, 5)]
impropers = []
partial_charges = [-0.4, 0.1, 0.1, 0.1, 0.1]
atomic_numbers = [6, 1, 1, 1, 1]

methane = Molecule(atoms=atoms, bonds=bonds, impropers=impropers, partial_charges=partial_charges, atomic_numbers=atomic_numbers)

# %%
# lets assume that you have the following data:

xyz = np.random.randn(25, 5, 3) # 25 states, 5 atoms, 3 dimensions
energy = np.random.randn(25) # 25 states
nonbonded_energy = np.random.randn(25) # 25 states

# then you can create a MolData object as follows:
moldata = MolData.from_arrays(xyz=xyz, energy=energy, nonbonded_energy=nonbonded_energy, molecule=methane)

# now the reference energies are computed as difference between qm and nonbonded. Then, they are centered.
print(moldata.energy_ref.shape)

#%%

# this allows you to obtain a dgl graph with the data written in there, such that the grappa model is directly applicable:

g = moldata.to_dgl()

print(g.nodes['g'].data['energy_ref'].shape)
# %%

# for efficient loading, we could create a grappa.data.Dataset and then a grappa.GraphDataLoader from the MolData object.

"""
Now we want to evaluate a model on the data. For this, grappa has the ExplicitEvaluator class, which can differentiate between the different dataset types.
"""

# lets define three moldata object. (Here, just copy the same data three times)

moldata2 = MolData.from_arrays(xyz=xyz, energy=energy, nonbonded_energy=nonbonded_energy, molecule=methane)

moldata3 = MolData.from_arrays(xyz=xyz, energy=energy, nonbonded_energy=nonbonded_energy, molecule=methane)

# now we pretend that the molecules are of different categories, which we call e.g. 'small molecules', or 'peptide'.
# we create two lists of same length. One with the dgl graphs that correspond to the moldata objects and one with the categories.

graphs = [moldata.to_dgl(), moldata2.to_dgl(), moldata3.to_dgl()]
ds_names = ['small molecules', 'peptide', 'small molecules']

#%%
# lets also load a model:
import torch
from grappa.utils.loading_utils import load_model

url = 'https://github.com/LeifSeute/test_torchhub/releases/download/test_release/protein_test_11302023.pth'

model = load_model(url)
# for predicting energies, we need to chain the model with an energy module:
class ParamFixer(torch.nn.Module):
    def forward(self, g):
        g.nodes['n2'].data['k'] = g.nodes['n2'].data['k'][:,0]
        g.nodes['n2'].data['eq'] = g.nodes['n2'].data['eq'][:,0]
        g.nodes['n3'].data['k'] = g.nodes['n3'].data['k'][:,0]
        g.nodes['n3'].data['eq'] = g.nodes['n3'].data['eq'][:,0]
        return g

# then, we can add the energy calculation module
from grappa.models.energy import Energy

model = torch.nn.Sequential(
    model,
    ParamFixer(),
    Energy(),
)


#%%
# create the evaluator:
from grappa.training.evaluation import ExplicitEvaluator

evaluator = ExplicitEvaluator(keep_data=True)

for g, dsname in zip(graphs, ds_names):
    with torch.no_grad():
        g = model(g)
    evaluator.step(g, [dsname])

metric_dict = evaluator.pool()
import json
print(json.dumps(metric_dict, indent=4))
print("Note that this is random mock data, so the metrics are not meaningful.")
# NOTE: THERE IS A PROBLEM WITH UNBATCH OR THE EVALUATOR. IF YOU ENCOUNTER 'RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation...', JUST REINITIALIZE THE GRAPHS
#%%
# You can obtain the individual energies as follows:
grappa_energies = evaluator.energies
small_mol_energies = grappa_energies['small molecules']
peptide_energies = grappa_energies['peptide']

print(small_mol_energies.shape)
# %%
ref_energies = evaluator.reference_energies
small_mol_energies_ref = ref_energies['small molecules']
print(small_mol_energies_ref.shape)
# %%
