#%%
"""
Evaluate a grappa dataset for a model.
"""
from grappa.data import Dataset

#%%
DSNAME = 'AA_natural'
# Download a dataset if not present already:
dataset = Dataset.from_tag(DSNAME)

dataset = dataset.where(['energy_amber99sbildn' in g.nodes['g'].data for g in dataset.graphs])

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
model = model.to(device)

#%%
# Grappa provides an evaluator class that collects data across batches and calculates metrics for the whole dataset.
from grappa.training.evaluation import ExplicitEvaluator
evaluator = ExplicitEvaluator(keep_data=True, suffix_classical='_amber99sbildn', log_classical_values=True, ref_suffix_classical='_qm')

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

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

for i, dsname in enumerate([DSNAME]):
    # Assuming evaluator and metrics objects are already defined and hold the necessary data for each dsname
    grads = evaluator.gradients[dsname].cpu().numpy()
    ref_grads = evaluator.reference_gradients[dsname].cpu().numpy()

    energies = evaluator.energies[dsname].cpu().numpy()
    ref_energies = evaluator.reference_energies[dsname].cpu().numpy()

    # Gradient plot for the dataset
    axs[0].scatter(ref_grads, grads)
    axs[0].set_xlabel('QM')
    axs[0].set_ylabel('Grappa')
    axs[0].set_title(f'{dsname} Gradient Components [kcal/mol/Å]')
    axs[0].text(0.1, 0.9, f'Component RMSE: {metrics[dsname]["crmse_gradients"]:.1f} kcal/mol/Å', 
                   transform=axs[0].transAxes)

    # Energy plot for the dataset
    axs[1].scatter(ref_energies, energies)
    axs[1].set_xlabel('QM')
    axs[1].set_ylabel('Grappa')
    axs[1].set_title(f'{dsname} Energies [kcal/mol]')

    axs[1].text(0.1, 0.9, f'RMSE: {metrics[dsname]["rmse_energies"]:.1f} kcal/mol', 
                   transform=axs[1].transAxes)

plt.tight_layout()  # Adjust layout
plt.savefig('evaluation.png')
plt.show()
# %%
g.nodes['g'].data.keys()
# %%
