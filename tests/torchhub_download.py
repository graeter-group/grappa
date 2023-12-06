#%%
from grappa.utils.run_utils import load_model

url = 'https://github.com/LeifSeute/test_torchhub/releases/download/test_release/protein_test_11302023.pth'

#%%
model = load_model(url)

# add an energy calculation module:
from grappa.models.energy import Energy
import torch

class ParamFixer(torch.nn.Module):
    def forward(self, g):
        g.nodes['n2'].data['k'] = g.nodes['n2'].data['k'][:,0]
        g.nodes['n2'].data['eq'] = g.nodes['n2'].data['eq'][:,0]
        g.nodes['n3'].data['k'] = g.nodes['n3'].data['k'][:,0]
        g.nodes['n3'].data['eq'] = g.nodes['n3'].data['eq'][:,0]
        return g

model = torch.nn.Sequential(
    model,
    ParamFixer(),
    Energy(suffix=''),
)

#%%
################
# Get a dataset
from grappa.data import Dataset, GraphDataLoader
from grappa.training.evaluation import ExplicitEvaluator
from grappa.utils.run_utils import get_data_path

dspath = get_data_path()/'peptides'/'dgl_datasets'/'tripeptides'

# ds = Dataset.load(dspath)
ds = Dataset.load(dspath)
loader = GraphDataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
#%%
evaluator = ExplicitEvaluator(suffix='', suffix_ref='_ref')
for g, dsname in loader:
    g = model(g)
    evaluator.step(g, dsname)
d = evaluator.pool()
print(d)
# %%
