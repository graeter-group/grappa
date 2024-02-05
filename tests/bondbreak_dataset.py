#%%
from grappa.data import Dataset

TAG = 'hyp-dop_amber99sbildn'
TAG = 'AA_bondbreak_rad'

ds = Dataset.from_tag(TAG)
# %%

qm_grads = []
ff_grads = []
for i in range(len(ds)):
    g, dsname = ds[i]
    qm_grads.append(g.nodes['n1'].data['gradient_qm'].detach().numpy())
    # ff_grads.append(g.nodes['n1'].data['gradient_reference_ff'].detach().numpy())

# %%
import numpy as np
import matplotlib.pyplot as plt

crmses = []
for i in range(len(ds)):
    # crmses.append(np.sqrt(np.mean((qm_grads[i] - ff_grads[i])**2)))
    crmses.append(np.std(qm_grads[i]))

plt.figure()
plt.hist(crmses, bins=50)
plt.show()
# %%
