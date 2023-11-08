#%%
import dgl
from pathlib import Path
import torch
import numpy as np
from to_npz import load_graph, load_mol, extract_data
import json
from create_split import make_smile_dict, get_all_duplicates
import copy

#%%



def main(dspath, targetpath):
    print(f"adding duplicates from\n{dspath}\nto sets in\n{targetpath}")
    dspath = Path(dspath)
    targetpath = Path(targetpath)

    assert targetpath.exists()

    dsnames = [p.name for p in targetpath.iterdir()]

    # iterate over all child directories of dspath:
    num_total = 0
    num_success = 0
    num_err = 0

    total_mols = 0

    duplicates = [] # list of lists of tuples of ds_name and idx of molecules that are duplicates across different datasets

    for idx, subdir in enumerate(dspath.iterdir()):
        if not subdir.is_dir():
            continue

        this_entry = []

        for subsubdir in subdir.iterdir():
            if not subsubdir.is_dir():
                continue
            dsname = subsubdir.name
            if not dsname in dsnames:
                raise ValueError(f"Dataset {dsname} not found in {targetpath}")
            
            for molpath in subsubdir.iterdir():
                # now we are at molpath=some number that was missing in the original dataset

                num_total += 1
                if not molpath.is_dir():
                    continue
            
                print(f"Processing {idx}", end='\r')
                g, mol = load_graph(molpath), load_mol(molpath)
                data = extract_data(g, mol)

                total_mols += 1

                dsfolder = targetpath/dsname
                assert dsfolder.exists()        

                # store the idx and dsname of the current molecule.                            
                this_entry.append((dsname, molpath.stem))

                np.savez_compressed(dsfolder/(molpath.stem+'.npz'), **data)
                num_success += 1

        # check whether different ds names occur in this entry. If they are all the same, we must not add it to the duplicates list as it only occurs once in our dataset (under id idx1-idx2-...)
        if len(set([t[0] for t in this_entry])) > 1:
            duplicates.append(this_entry)

    print("\nDone!")
    print(f"Processed {num_total} molecules, {num_success} successfully, {num_err} with errors")

    with open(targetpath/"duplicates_espaloma.json", 'w') as file:
        json.dump(duplicates, file, indent=4)

    duplicates_espaloma = copy.deepcopy(duplicates)

    # these duplicates are incomplete: there are more molecules with the same smiles string:
    smiledict = make_smile_dict(targetpath)
    np.savez_compressed(targetpath/'smiles.npz', **smiledict)
    duplicates = get_all_duplicates(smile_dict=smiledict)

    with open(targetpath/'duplicates.json', 'w') as f:
        json.dump(duplicates, f, indent=4)

    print(f"Duplicates recognized by espaloma: {len(duplicates_espaloma)}")

    print(f"Actual duplicate smiles: {len(duplicates)}")


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