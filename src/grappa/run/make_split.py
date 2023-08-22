#%%

import argparse
from pathlib import Path
from grappa.constants import DEFAULTBASEPATH, DS_PATHS
import json
from grappa.PDBData.PDBDataset import PDBDataset, SplittedDataset
import os


def make_split():
    parser = argparse.ArgumentParser(description='Create split names for k-fold cross validation.')
    parser.add_argument('fold_path', type=str, default="", help='Folder to save the folds in.')
    parser.add_argument('--ds_short', type=str, nargs="+", default=[], help='Short names of datasets.')
    parser.add_argument('--k', type=int, default=5, help='Number of folds.')
    parser.add_argument('--ds_path', type=str, nargs="+", default=[], help='Paths to datasets.')

    args = parser.parse_args()

    for short in args.ds_short:
        assert short in DS_PATHS.keys(), f"Unknown short name {short}."
        dspath = DS_PATHS[short]
        if not dspath in args.ds_path:
            args.ds_path.append(dspath)
    
    assert not len(args.ds_path) == 0, "No dataset paths given."

    datasets = [PDBDataset.load_npz(path) for path in args.ds_path]
    folds = SplittedDataset.do_kfold_split(datasets, k=args.k)

    os.makedirs(args.fold_path, exist_ok=True)

    for i, fold in enumerate(folds):
        with open(os.path.join(args.fold_path,f"fold_{i}.json"), "w") as f:
            json.dump(fold, f, indent=4)