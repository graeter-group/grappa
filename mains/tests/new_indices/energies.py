#%%
from grappa.run import run_utils
from grappa.ff_utils.create_graph import utils, tuple_indices

p = "/hits/fast/mbm/seutelf/data/datasets/old_PDBDatasets/spice/amber99sbildn_amber99sbildn_60_dgl.bin"
[ds], _ = run_utils.get_data([p], 10)
# %%

import numpy as np
torsions = np.array([])
impropers = np.array([])

for g in ds:
    torsions = np.concatenate((torsions, g.nodes["g"].data["u_torsion_amber99sbildn"].numpy().flatten()))
    impropers = np.concatenate((impropers, g.nodes["g"].data["u_improper_amber99sbildn"].numpy().flatten()))


# print mean and std dev:
print(f"torsions: mean: {np.mean(torsions)}, std: {np.std(torsions)}")
print(f"impropers: mean: {np.mean(impropers)}, std: {np.std(impropers)}")
# %%
import matplotlib.pyplot as plt
# do a plot to compare the probability distributions for torsions and impropers

plt.figure(figsize=(10, 6))

plt.hist(torsions, bins=10, alpha=0.5, label='Torsions', density=True)
plt.hist(impropers, bins=10, alpha=0.5, label='Impropers', density=True)

plt.title('Probability Distributions')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()

plt.show()
# %%
