#%%
import numpy as np
from grappa.data import MolData
from pathlib import Path

#%%
p = '/hits/fast/mbm/hartmaec/workdir/new_dataset/dataset_clean/AA_break/eq_average'
p = Path(p)

for moldir in p.iterdir():
    if not moldir.is_dir():
        continue
    molname = moldir.name
    moldata = MolData.load(moldir/'moldata.npz')
    break
print(len(moldata.energy_ref))
print(moldata.molecule.additional_features['is_radical'])

# %%
print(moldata.energy - moldata.energy.min())
# %%
