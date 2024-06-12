#%%
import matplotlib.pyplot as plt
from grappa.utils.plotting import scatter_plot
import numpy as np

# %%
datapath = "/local/user/seutelf/grappa/ckpt/grappa-1.3/amber99-bonds-angles-torsions/2024-06-11_14-40-54/test_data/epoch-309.npz"

data = np.load(datapath)

list(data.keys())
# %%
gradients = data["gradients:dipeptides-300K-amber99"].flatten()
gradients_ref = data["reference_gradients:dipeptides-300K-amber99"].flatten()

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
scatter_plot(ax, gradients, gradients_ref, cluster=True, logscale=True)
# %%
datapath = "/local/user/seutelf/grappa/ckpt/grappa-1.3/amber99-bonds-angles-torsions/2024-06-11_14-40-54/test_data/amber99sbildn/data.npz"

data = np.load(datapath)

list(data.keys())
# %%
gradients = data["gradients:dipeptides-300K-amber99"].flatten()
gradients_ref = data["reference_gradients:dipeptides-300K-amber99"].flatten()

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
scatter_plot(ax, gradients, gradients_ref, cluster=True, logscale=True)
# %%
