#%%
from grappa.data import MolData
from grappa.wrappers import openmm_wrapper
from grappa.utils import loading_utils
from grappa.models import GrappaModel, get_default_model_config, model_from_config

#%%

# model = loading_utils.model_from_tag('grappa-1.0').cpu()

config = get_default_model_config()
config['torsion_cutoff'] = 1e-4

model = model_from_config(config).cpu()
# %%
# load the dgl datasets:
from grappa.data import Dataset

ds = Dataset.from_tag('spice-des-monomers')

proper_ks = []
improper_ks = []

for i, (g, dsname) in enumerate(ds):
    if i > 100:
        break
    print(i, end='\r')
    g = model(g)
    proper_ks.append(g.nodes['n4'].data['k'].detach().flatten().cpu().numpy())

    improper_ks.append(g.nodes['n4_improper'].data['k'].detach().flatten().cpu().numpy())

#%%
    
import numpy as np
import matplotlib.pyplot as plt

# make a histogram of the abs vals in logscale k

proper_ks = np.concatenate(proper_ks)
improper_ks = np.concatenate(improper_ks)

# remove the zeros:
proper_ks = proper_ks[proper_ks != 0]
improper_ks = improper_ks[improper_ks != 0]

#%%

plt.hist(np.log10(np.abs(proper_ks)), bins=100, alpha=0.5, label='proper')

plt.hist(np.log10(np.abs(improper_ks)), bins=100, alpha=0.5, label='improper')

plt.legend()
# %%
