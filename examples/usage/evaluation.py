#%%
"""
Evaluate a grappa dataset for a model.
"""
from grappa.data import Dataset

#%%
DSNAME = 'spice-dipeptide_amber99sbildn'
DSNAME2 = 'dipeptide_rad'
DSNAME3 = 'spice-des-monomers'

# Download a dataset if not present already:
dataset = Dataset.from_tag(DSNAME)
# Download a second dataset and append it:
dataset += Dataset.from_tag(DSNAME2)
dataset += Dataset.from_tag(DSNAME3)

dataset.remove_uncommon_features()

# For more efficient data loading, we use a GraphDataLoader
from grappa.data import GraphDataLoader

loader = GraphDataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
# %%

# Download a model if not present already:
from grappa.utils.loading_utils import model_from_tag

model = model_from_tag('latest')

#%%
# add an energy calculation module
import torch
from grappa.models.energy import Energy

model = torch.nn.Sequential(
    model,
    Energy(),
)

#%%

# Push the model to the GPU if available:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
model = model.to(device)

#%%
# Grappa provides an evaluator class that collects data across batches and calculates metrics for the whole dataset.
from grappa.training.evaluation import Evaluator
evaluator = Evaluator(keep_data=True)

# Now, we can evaluate the model on the dataset:
for g, dsname in loader:
    g = g.to(device)
    g = model(g)
    evaluator.step(g, dsname)
#%%
# Finally, we can print the results:
metrics = evaluator.pool()

import json
print(json.dumps(metrics, indent=4))

# %%
# we can also plot the datapoints:
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 2, figsize=(10, 15))

for i, dsname in enumerate([DSNAME, DSNAME2, DSNAME3]):
    # Assuming evaluator and metrics objects are already defined and hold the necessary data for each dsname
    grads = evaluator.all_gradients[dsname].cpu().numpy()
    ref_grads = evaluator.all_reference_gradients[dsname].cpu().numpy()

    energies = evaluator.all_energies[dsname].cpu().numpy()
    ref_energies = evaluator.all_reference_energies[dsname].cpu().numpy()

    # Gradient plot for the dataset
    axs[i, 0].scatter(ref_grads, grads)
    axs[i, 0].set_xlabel('QM')
    axs[i, 0].set_ylabel('Grappa')
    axs[i, 0].set_title(f'{dsname} Gradient Components [kcal/mol/Å]')
    axs[i, 0].text(0.1, 0.9, f'Component RMSE: {metrics[dsname]["crmse_gradients"]:.1f} kcal/mol/Å', 
                   transform=axs[i, 0].transAxes)

    # Energy plot for the dataset
    axs[i, 1].scatter(ref_energies, energies)
    axs[i, 1].set_xlabel('QM')
    axs[i, 1].set_ylabel('Grappa')
    axs[i, 1].set_title(f'{dsname} Energies [kcal/mol]')
    # Assuming you have a similar RMSE metric for energies
    axs[i, 1].text(0.1, 0.9, f'RMSE: {metrics[dsname]["rmse_energies"]:.1f} kcal/mol', 
                   transform=axs[i, 1].transAxes)

plt.tight_layout()  # Adjust layout
plt.savefig('evaluation.png')
plt.show()
# %%

from grappa.data import MolData
from grappa.data import Molecule

md = MolData.load('/hits/fast/mbm/seutelf/grappa/data/grappa_datasets/tripeptides_amber99sbildn/0.npz')

g = md.to_dgl()

print(g.nodes['n1'].data['charge_model'])
# %%
out = model(g)

grads = out.nodes['n1'].data['gradient'].flatten().detach().cpu().numpy()
grads_ref = out.nodes['n1'].data['gradient_ref'].flatten().detach().cpu().numpy()

plt.scatter(grads_ref, grads)
# %%
from openmm.app import PDBFile, ForceField

pdb = PDBFile('T4.pdb')

system = ForceField('amber99sbildn.xml', 'tip3p.xml').createSystem(pdb.topology)

mol = Molecule.from_openmm_system(system, pdb.topology, charge_model='classical')

molg = mol.to_dgl()

print(molg.nodes['n1'].data['charge_model'][0:5])
# %%
