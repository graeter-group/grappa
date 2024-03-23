#%%
"""
By copying the procedure from espaloma, creates a list of smiles strings for the validation and test set of the espaloma dataset split. These smiles strings are used as ids for splitting the dataset in grappa.
"""
# from https://github.com/choderalab/refit-espaloma/tree/main, 10.01.2024
# https://github.com/choderalab/refit-espaloma/blob/main/openff-default/02-train/joint-improper-charge/charge-weight-1.0/train.py
import espaloma as esp
import os
import random
import glob
# %%
DATASETS = ["gen2", "gen2-torsion", "pepconf-dlc", "protein-torsion", "spice-pubchem", "spice-dipeptide", "spice-des-monomers", "rna-nucleoside", "rna-diverse"]

RANDOM_SEED = 2666
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1


#-------------------------
# SUBMODULE
#-------------------------
def _load_datasets(datasets, input_prefix):
    """
    Load unique molecules (nonisomeric smile).
    """
    for i, dataset in enumerate(datasets):
        path = os.path.join(input_prefix, dataset)
        ds = esp.data.dataset.GraphDataset.load(path).shuffle(RANDOM_SEED)
        _ds_tr, _ds_vl, _ds_te = ds.split([TRAIN_RATIO, VAL_RATIO, TEST_RATIO])

        # Merge datasets
        if i == 0:
            ds_vl = _ds_vl
            ds_te = _ds_te
        else:
            ds_vl += _ds_vl
            ds_te += _ds_te
    del ds, _ds_tr, _ds_vl, _ds_te

    #
    # Load duplicated molecules
    #
    print("# LOAD DUPLICATED MOLECULES")
    entries = glob.glob(os.path.join(input_prefix, "duplicated-isomeric-smiles-merge", "*"))
    random.seed(RANDOM_SEED)
    random.shuffle(entries)

    n_entries = len(entries)
    entries_tr = entries[:int(n_entries*TRAIN_RATIO)]
    entries_vl = entries[int(n_entries*TRAIN_RATIO):int(n_entries*TRAIN_RATIO)+int(n_entries*VAL_RATIO)]
    entries_te = entries[int(n_entries*TRAIN_RATIO)+int(n_entries*VAL_RATIO):]
    print("Found {} entries. Split data into {}:{}:{} entries.".format(n_entries, len(entries_tr), len(entries_vl), len(entries_te)))
    assert n_entries == len(entries_tr) + len(entries_vl) + len(entries_te)

    for entry in entries_vl:
        _datasets = os.listdir(entry)
        for _dataset in _datasets:
            _ds_vl = esp.data.dataset.GraphDataset.load(os.path.join(entry, _dataset))
            ds_vl += _ds_vl
    for entry in entries_te:
        _datasets = os.listdir(entry)
        for _dataset in _datasets:
            _ds_te = esp.data.dataset.GraphDataset.load(os.path.join(entry, _dataset))
            ds_te += _ds_te

    print(f"The final validate and test data size is {len(ds_vl)} and {len(ds_te)}.")
    return ds_vl, ds_te


# %%
from pathlib import Path

input_prefix = Path(__file__).parent.parent.parent/'data/esp_data'
assert input_prefix.exists()
# %%
ds_vl, ds_te = _load_datasets(DATASETS, input_prefix)

vl_smiles = set([mol.mol.to_smiles() for mol in ds_vl])
te_smiles = set([mol.mol.to_smiles() for mol in ds_te])

import json

with open('vl_smiles.json', 'w') as f:
    json.dump(list(vl_smiles), f)
with open('te_smiles.json', 'w') as f:
    json.dump(list(te_smiles), f)