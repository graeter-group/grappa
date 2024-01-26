#%%
import dgl
from pathlib import Path
import torch
import numpy as np
from to_npz import load_graph, load_mol, extract_data
import json
import copy
import shutil

#%%



def main(dspath, targetpath):
    print(f"adding duplicates from\n{dspath}\nto sets in\n{targetpath}")
    dspath = Path(dspath)
    targetpath = Path(targetpath)

    assert targetpath.exists()

    dsnames = [p.name for p in targetpath.iterdir() if p.is_dir()]
    assert len(list(set(dsnames))) == len(dsnames), f"Duplicate dataset names in targetpath: {dsnames}"

    # iterate over all child directories of the duplicates path
    num_total = 0
    num_already_there = 0

    # subdir: 0, 1, 2, ... in the duplicates path
    for idx, subdir in enumerate(dspath.iterdir()):
        if not subdir.is_dir():
            continue

        # subsubdir: 0/dsname1, 0/dsname2, ...
        for subsubdir in subdir.iterdir():
            if not subsubdir.is_dir():
                continue
            dsname = subsubdir.name
            if not dsname in dsnames:
                raise ValueError(f"Dataset {dsname} not found in {targetpath}")
            
            assert len(list(subsubdir.iterdir())) == 1, f"subsubdir {subsubdir} has more than one child, should only have one entry: one molecule that occured several times in the original dataset."
            
            print(f"Processing {idx}", end='\r')
            
            # now copy the whole content of {n}/dsname to targetpath/dsname:
            dsfolder = targetpath/dsname
            assert dsfolder.exists()        

            # copy:
            for p in subsubdir.iterdir():
                if p.name in [p.name for p in dsfolder.iterdir()]:
                    num_already_there += 1
                else:
                    # copy recursively:
                    shutil.copytree(p, dsfolder/p.name)
                    num_total += 1



    print("\nDone!")
    print(f"Copied {num_total} molecules, skipped {num_already_there} because they were already there.")


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add duplicates back to their datasets. We can do this because we use the smiles string for differentiating between molecules regardless of their dataset origin in our dataset splitting routine.")
    parser.add_argument(
        "--duplpath",
        type=str,
        default="/hits/fast/mbm/seutelf/esp_data/duplicated-isomeric-smiles-merge",
        help="Path to the folder with heterograph and mol files from espaloma.",
    )
    parser.add_argument(
        "--targetpath",
        type=str,
        default="/hits/fast/mbm/seutelf/data/esp_data",
        help="Path to a folder containingn the raw espaloma datasets.",
    )

    args = parser.parse_args()
    main(dspath=args.duplpath, targetpath=args.targetpath)