#%%
from grappa.data import Dataset, GraphDataLoader
from grappa.training.evaluation import ExplicitEvaluator
from grappa.utils.dataset_utils import get_data_path
from grappa.training import config
#%%

#%%
# dspath = get_data_path()/'peptides'/'dgl_datasets'/'spice'
# dspath = get_data_path()/'dgl_datasets'/'spice-des-monomers'
# ds = Dataset.load(dspath)
dspaths = config.default_config()['data_config']['datasets']
ds = Dataset()
for dspath in dspaths:
    ds += Dataset.load(dspath)
#%%
# dspath = get_data_path()/'dgl_datasets'/'protein-torsion'
# ds = Dataset.load(dspath)
loader = GraphDataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
evaluator = ExplicitEvaluator(suffix='_reference_ff', suffix_ref='_qm')
evaluator = ExplicitEvaluator(suffix='_gaff-2.11', suffix_ref='_qm')
for g, dsname in loader:
    evaluator.step(g, dsname)
# %%
d = evaluator.pool()
d
# %%
len(loader)
# %%
dspath = get_data_path()/'dgl_datasets'/'spice-des-monomers'
ds = Dataset.load(dspath)