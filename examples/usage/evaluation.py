#%%
"""
Evaluate a grappa dataset for a model.
"""
from grappa.data import Dataset

#%%
# Download a dataset if not present already:
dataset = Dataset.from_tag('tripeptides_amber99sbildn')

# For more efficient data loading, we use a GraphDataLoader
from grappa.data import GraphDataLoader

loader = GraphDataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
# %%

# Download a model if not present already:
from grappa.utils.loading_utils import load_model

url = 'https://github.com/LeifSeute/test_torchhub/releases/download/test_release/protein_test_11302023.pth'

model = load_model(url)

#%%
# add an energy calculation module

# first, we have to add a module that fixes the output shape of the model (this will not be necessary in the future)

import torch

class ParamFixer(torch.nn.Module):
    def forward(self, g):
        g.nodes['n2'].data['k'] = g.nodes['n2'].data['k'][:,0]
        g.nodes['n2'].data['eq'] = g.nodes['n2'].data['eq'][:,0]
        g.nodes['n3'].data['k'] = g.nodes['n3'].data['k'][:,0]
        g.nodes['n3'].data['eq'] = g.nodes['n3'].data['eq'][:,0]
        return g

# then, we can add the energy calculation module
from grappa.models.Energy import Energy

model = torch.nn.Sequential(
    model,
    ParamFixer(),
    Energy(),
)

#%%

# Push the model to the GPU if available:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#%%
# Grappa provides an evaluator class that collects data across batches and calculates metrics for the whole dataset.
from grappa.training.evaluation import ExplicitEvaluator
evaluator = ExplicitEvaluator(keep_data=True)

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

dsname = 'tripeptides_amber99sbildn'

plt.scatter(evaluator.gradients[dsname].cpu().numpy(), evaluator.reference_gradients[dsname].cpu().numpy())
plt.xlabel('Grappa')
plt.ylabel('QM')
plt.title('Gradient Components [kcal/mol/Å]')
plt.text(0.1, 0.9, f'Component RMSE: {metrics[dsname]["crmse_gradients"]:.1f} kcal/mol/Å', transform=plt.gca().transAxes)
plt.show()
# %%
