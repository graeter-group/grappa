#%%
import dgl
from pathlib import Path
import torch
import numpy as np
from to_npz import load_graph, load_mol, extract_data
import json
import copy

#%%



def main(dspath, targetpath):
    print(f"adding duplicates from\n{dspath}\nto sets in\n{targetpath}")
    dspath = Path(dspath)
    targetpath = Path(targetpath)

    assert targetpath.exists()

    dsnames = [p.name for p in targetpath.iterdir() if p.is_dir()]
    assert len(list(set(dsnames))) == len(dsnames), f"Duplicate dataset names in targetpath: {dsnames}"

    # iterate over all child directories of dspath (this is the duplicates path)
    num_total = 0
    num_success = 0
    num_err = 0

    total_mols = 0

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
            molpath = next(iter(subsubdir.iterdir()))
            # now we are at molpath=string with some numbers that were missing in the original dataset. store this as npz in the targetpath/dsname folder

            num_total += 1

            print(f"Processing {idx}", end='\r')
            g, mol = load_graph(molpath), load_mol(molpath)
            data = extract_data(g, mol)

            total_mols += 1

            dsfolder = targetpath/dsname
            assert dsfolder.exists()        

            # store the idx and dsname of the current molecule.                            

            np.savez_compressed(dsfolder/(molpath.stem+'.npz'), **data)
            num_success += 1

    print("\nDone!")
    print(f"Processed {num_total} molecules, {num_success} successfully, {num_err} with errors")


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duplpath",
        type=str,
        default="/hits/fast/mbm/seutelf/esp_data/duplicated-isomeric-smiles-merge",
        help="Path to the folder with heterograph and mol files from espaloma.",
    )
    parser.add_argument(
        "--targetpath",
        type=str,
        default="/hits/fast/mbm/seutelf/data/datasets",
        help="Path to a folder containing containing the datasets to which the duplicates will be added.",
    )

    args = parser.parse_args()
    main(dspath=args.duplpath, targetpath=args.targetpath)