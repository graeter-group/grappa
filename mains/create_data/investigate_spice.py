#%%
from grappa.PDBData.PDBDataset import PDBDataset, SplittedDataset
from grappa.constants import DEFAULTBASEPATH
from pathlib import Path

#%%
ds = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"monomers"/"charge_default_ff_gaff-2_11", n_max=None)

# %%
ds2 = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"qca_spice"/"charge_default_ff_gaff-2_11", n_max=None)

#%%
ds3 = PDBDataset.load_npz(Path(DEFAULTBASEPATH)/"pubchem"/"charge_default_ff_gaff-2_11", n_max=None)
# %%
ds.evaluate(plotpath="monomer_plots_unfiltered", by_element=True, by_residue=False, suffix="_total_ref", radicals=False)
# %%
ds2.evaluate(plotpath="qca_spice_plots_unfiltered", by_element=True, by_residue=False, suffix="_total_ref", radicals=False)
#%%
ds3.evaluate(plotpath="pubchem_plots_unfiltered", by_element=True, by_residue=False, suffix="_total_ref", radicals=False)
# %%
ds.filter_confs(max_energy=65, max_force=200, reference=False)
ds2.filter_confs(max_energy=65, max_force=200, reference=False)
ds3.filter_confs(max_energy=65, max_force=200, reference=False)
# %%
ds.evaluate(plotpath="monomer_plots", by_element=True, by_residue=False, suffix="_total_ref", radicals=False)
ds2.evaluate(plotpath="qca_spice_plots", by_element=True, by_residue=False, suffix="_total_ref", radicals=False)
#%%
ds3.evaluate(plotpath="pubchem_plots", by_element=True, by_residue=False, suffix="_total_ref", radicals=False)
#%%

#################
# just some testing of split functionality
#################

# # find the molecule with name disulfide:
# mol = None
# for mol_ in ds2.mols:
#     if mol_.name.lower() == "disulfide":
#         mol = mol_
#         break
# print(mol.name)
# # %%
# def get_rmse(mol):
#     en = mol.energies
#     en_gaff = mol.graph_data["g"]["u_total_ref"][0]
#     en -= en.mean()
#     en_gaff -= en_gaff.mean()
#     import numpy as np

#     return np.sqrt(np.mean((en - en_gaff)**2))

# get_rmse(mol)
# # %%

# seed = 0

# ds_splitter = SplittedDataset.create([ds2,ds], [0.8, 0.1, 0.1], seed=seed)

# # %%
# vl_ds2, vl_ds = ds_splitter.get_splitted_datasets([ds2,ds], ds_type="vl")
# # %%
# rmses = list(map(get_rmse, vl_ds2.mols))
# #%%
# seed = 0
# ds_splitter = SplittedDataset.create([ds2,ds], [0.8, 0.1, 0.1], seed=seed)

# # %%
# vl_ds2, vl_ds = ds_splitter.get_splitted_datasets([ds2,ds], ds_type="vl")

# # %%
# rmses == list(map(get_rmse, vl_ds2.mols))
# # %%
