#%%
from grappa.PDBData.PDBDataset import PDBDataset, SplittedDataset
from grappa.constants import DS_PATHS
import json

#%%
ds1 = PDBDataset.load_npz(DS_PATHS["collagen"], n_max=None)
ds2 = PDBDataset.load_npz(DS_PATHS["spice"], n_max=None)
ds3 = PDBDataset.load_npz(DS_PATHS["radical_dipeptides"], n_max=None)

# %%
folds = SplittedDataset.do_kfold_split([ds1, ds2, ds3], k=5)
# %%

assert len(folds) == 5
assert len(folds[2]) == 3

# no intersection between train and test
assert len(list(set(folds[2][0]).intersection(set(folds[2][1])))) == 0
# no intersection between test of fold 0 and test of fold 2
assert len(list(set(folds[0][2]).intersection(set(folds[2][2])))) == 0
# %%
for i, fold in enumerate(folds):
    with open(f"fold_{i}.json", "w") as f:
        json.dump(fold, f, indent=4)
# %%

ds4 = PDBDataset.load_npz(DS_PATHS["tripeptides"], n_max=None)
#%%
splitter = SplittedDataset.create_with_names([ds1, ds2, ds3, ds4], split=[1, 0, 0], split_names=folds[4])
#%%
print(splitter.split_names)
#%%
tr_loader, vl_loader, te_loader = splitter.get_full_loaders()
# %%
print(len(tr_loader), len(vl_loader), len(te_loader))
# %%
splitter = SplittedDataset.create_with_names([ds1, ds2, ds3, ds4], split=[1, 0, 0], split_names=folds[2])

tr_loader, vl_loader, te_loader = splitter.get_full_loaders()
# %%
print(len(tr_loader), len(vl_loader), len(te_loader))
# %%

tested_mols = 0
for fold in folds:
    splitter = SplittedDataset.create_with_names([ds1, ds2, ds3, ds4], split=[1, 0, 0], split_names=fold)
    tr_loader, vl_loader, te_loader = splitter.get_full_loaders()
    tested_mols += len(te_loader)

assert tested_mols == len(ds1) + len(ds2) + len(ds3), f"tested {tested_mols} mols, but should have tested {len(ds1) + len(ds2) + len(ds3)} mols"
print("done")
# %%
