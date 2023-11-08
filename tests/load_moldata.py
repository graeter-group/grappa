#%%
from pathlib import Path
from grappa.data import MolData
import numpy as np

dspath = Path(__file__).parents[1]/'data'/"grappa_datasets"/"rna-nucleoside"
#%%
for file in dspath.iterdir():
    if file.suffix == '.npz':
        moldata = MolData.load(file)
# %%
