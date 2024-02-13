#%%
from grappa.data import MolData
from grappa.wrappers import openmm_wrapper
from grappa.utils import loading_utils

#%%

molpath1 = '/hits/fast/mbm/seutelf/grappa/data/grappa_datasets/spice-dipeptide_amber99sbildn/3.npz'

molpath2 = '/hits/fast/mbm/seutelf/grappa/data/grappa_datasets/spice-dipeptide/3.npz'

mol1 = MolData.load(molpath1)
mol2 = MolData.load(molpath2)

#%%

print(mol1.molecule.additional_features['charge_model'].mean(0))
print(mol2.molecule.additional_features['charge_model'].mean(0))
# %%
