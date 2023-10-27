import os
import json
import random
from pathlib import Path
import copy

import numpy as np

from collections import defaultdict


def get_all_duplicates(smile_dict):
    """
    Returns a list of lists of lists mol=[dsname, id] for mols with the same smiles string.
    """
    # Initialize a dictionary to store smiles as keys and lists of (dsname, id) as values.
    smiles_to_mols = defaultdict(list)

    print(f"checking for duplicates in dataset {[dsname for dsname in smile_dict.keys()]}...")
    
    # Populate the dictionary.
    for dsname, array in smile_dict.items():
        ids, smiles = array[0], array[1]
        for id, smile in zip(ids, smiles):
            smiles_to_mols[smile].append([dsname, id])
    
    # Filter out unique smiles and return only duplicates.
    duplicates = [smiles_to_mols[smile] for smile in smiles_to_mols.keys() if len(smiles_to_mols[smile]) > 1]
    
    return duplicates



def make_smile_dict(dspath: str) -> dict:
    dspath = Path(dspath)
    smile_dict = {}
    
    for ds_folder in dspath.iterdir():
        if not ds_folder.is_dir():
            continue

        dsname = ds_folder.name

        print(f"Creating smiles dict for {dsname}...")

        ids_list = []
        smiles_list = []
        
        for file in ds_folder.glob("*.npz"):
            data = np.load(file)
            smiles = data['smiles'].item()
            id = file.stem
            ids_list.append(id)
            smiles_list.append(smiles)
        
        smile_dict[dsname] = np.array([ids_list, smiles_list])
    
    return smile_dict


def load_smiles(dspath, splitpath, split_id=None):
    dspath = Path(dspath)
    splitpath = Path(splitpath)

    if (dspath/'smiles.npz').exists():
        # Load smiles from single file
        data = np.load(dspath/'smiles.npz')
        smiledict = {dsname: data[dsname] for dsname in data.keys()}
        print("Loaded smiles from single file")

    else:
        smiledict = make_smile_dict(dspath)
        np.savez_compressed(dspath/'smiles.npz', **smiledict)

    # convert to list:
    smiledict = {dsname: [list(smiledict[dsname][0]), list(smiledict[dsname][1])] for dsname in smiledict.keys()}


    train_smiles, val_smiles, test_smiles = [], [], []

    train_ds_ids, val_ds_ids, test_ds_ids = [], [], []

    for dsname in smiledict.keys():
        
        splitfile = (splitpath / dsname) / f"split{'_' + str(split_id) if split_id is not None else ''}.json"
        if not splitfile.exists():
            raise ValueError(f"Split file {splitfile} not found")
        
        with open(splitfile, 'r') as f:
            splits = json.load(f)

        print(f"Splitting smiles in {dsname}...")
        for split_type in ["train", "val", "test"]:
            for id in splits[split_type]:
                idx = smiledict[dsname][0].index(id)
                smiles = smiledict[dsname][1][idx]
                if split_type == "train":
                    train_smiles.append(smiles)
                    train_ds_ids.append((dsname, id))
                elif split_type == "val":
                    val_smiles.append(smiles)
                    val_ds_ids.append((dsname, id))
                elif split_type == "test":
                    test_smiles.append(smiles)
                    test_ds_ids.append((dsname, id))

    return (train_smiles, val_smiles, test_smiles), (train_ds_ids, val_ds_ids, test_ds_ids)





############################################################################################################


def create_split(dspath, splitpath, partition:(0.8, 0.1, 0.1), split_id=None, seed=0, check=False, create_duplicates=False):
    '''
    Create a dictionary of splits for each dataset in dspath. The splits are saved as split_{split_id}.json in splitpath/dataset_name.
    There has to be a duplicates.json file in dspath, which contains a list of lists of duplicates for each dataset.

    This function relies on the correctness of the duplicates.json file. If this file is not correct, the splits will be wrong.
    '''

    random.seed(seed)

    assert isinstance(partition, tuple) or isinstance(partition, dict), "Partition must be a tuple or dict"

    dspath = Path(dspath)
    
    if not create_duplicates:
        # Load duplicates list
        assert (dspath/'duplicates.json').exists(), f"Duplicates file not found at {dspath/'duplicates.json'}"
        with open(dspath/'duplicates.json', 'r') as f:
            duplicates = json.load(f)

    else:
        # Create duplicates list
        if (dspath/'smiles.npz').exists():
            # Load smiles from single file
            data = np.load(dspath/'smiles.npz')
            smiledict = {dsname: data[dsname] for dsname in data.keys()}
            print("Loaded smiles from single file")
        else:
            smiledict = make_smile_dict(dspath)
            np.savez_compressed(dspath/'smiles.npz', **smiledict)
        duplicates = get_all_duplicates(smile_dict=smiledict)
        with open(dspath/'duplicates.json', 'w') as f:
            json.dump(duplicates, f, indent=4)

    
    
    # First, split all duplicates. For this, use the partition that belongs to the first dataset in the partition dict that occurs in the duplicates list.

    if isinstance(partition, tuple):
        dupl_partition = partition
    else:
        for ds_name in sorted(partition.keys()):
            if ds_name in [ds for ds, idx in duplicates]:
                dupl_partition = partition[ds_name]
                break


    random.shuffle(duplicates)
    total_count = len(duplicates)
    
    n_train = int(total_count * dupl_partition[0])
    n_val = int(total_count * dupl_partition[1])
    n_test = total_count - n_train - n_val

    dup_train = copy.deepcopy(duplicates[:n_train])
    dup_val = copy.deepcopy(duplicates[n_train:n_train+n_val])
    dup_test = copy.deepcopy(duplicates[n_train+n_val:])

    assert len(dup_train) + len(dup_val) + len(dup_test) == total_count, f"Duplicates split failed, {len(dup_train) + len(dup_val) + len(dup_test)} != {total_count}"

    # flatten list axis 1 AFTER splitting
    dup_train = [mol for sublist in dup_train for mol in sublist]
    dup_val = [mol for sublist in dup_val for mol in sublist]
    dup_test = [mol for sublist in dup_test for mol in sublist]

    all_dups = dup_train + dup_val + dup_test

    # Iterate through datasets
    for ds_dir in dspath.iterdir():
        ds_name = ds_dir.name

        if not isinstance(partition, tuple):
            ds_partition = partition[ds_name]
        else:
            ds_partition = partition

        assert abs(sum(ds_partition) - 1) <= 1e-10, "Partitions must sum to 1.0"
        assert len(ds_partition) == 3, "Partitions must be a tuple of length 3"

        
        if not ds_dir.is_dir():
            continue

        # Initialize partitions
        train, val, test = [], [], []

        all_files = [f.stem for f in Path(ds_dir).glob('*.npz')]

        # transform to tuples:
        all_dups = [(ds_name, id) for ds_name, id in all_dups]

        # Generate list of all IDs without duplicates
        all_ids = [f.stem for f in Path(ds_dir).glob('*.npz') if (ds_name, f.stem) not in all_dups]

        print(len(all_ids), len(all_files))
        
        # Shuffle
        random.shuffle(all_ids)

        # Add duplicates to each partition
        train.extend([idx for ds, idx in dup_train if ds == ds_name])
        val.extend([idx for ds, idx in dup_val if ds == ds_name])
        test.extend([idx for ds, idx in dup_test if ds == ds_name])

        # Calculate remaining number of samples needed for each partition to reach partition size
        total_count = len(all_ids) + len(train) + len(val) + len(test)

        n_add_train = int(total_count * partition[0]) - len(train)
        n_add_val = int(total_count * partition[1]) - len(val)
        n_add_test = total_count - n_add_train - n_add_val - len(test)


        # Add the rest to each partition
        train.extend(all_ids[:n_add_train])
        val.extend(all_ids[n_add_train:n_add_train + n_add_val])
        test.extend(all_ids[n_add_train + n_add_val:])
        
        
        # Save to JSON
        split_dict = {'train': train, 'val': val, 'test': test}
        
        splitname = "split" if split_id is None else f"split_{split_id}"

        outpath = Path(splitpath)/ds_name/f'{splitname}.json'

        outpath.parent.mkdir(parents=True, exist_ok=True)

        with open(outpath, 'w') as f:
            json.dump(split_dict, f, indent=4)        

    if check:
        print("Checking split...")
        (tr, vl, te), (tr_ids, vl_ids, te_ids) = load_smiles(dspath=dspath, splitpath=splitpath, split_id=split_id)
        total_num = len(tr) + len(vl) + len(te)
        print(f"Total number of molecules: {total_num}")
        if total_num > 0:
            print(f"Molecules in train: {round(len(tr)/total_num*100, 2)}%")
            print(f"Molecules in val: {round(len(vl)/total_num*100, 2)}%")
            print(f"Molecules in test: {round(len(te)/total_num*100, 2)}%")


        print("Checking split of smiles strings...")
        # Check for overlapping smiles

        overlap_train_val = list(set(tr).intersection(set(vl)))
        if overlap_train_val:
            # find all occurences ((dsname, id), idx) of the overlapping smiles in the train and val sets
            # occurences = [(x, overlap_train_val.index(x[1])) for i, x in enumerate(tr_ids+vl_ids) if x[1] in overlap_train_val]
            occurences = [(x, overlap_train_val.index(smiles)) for x, smiles in zip(tr_ids+vl_ids, tr+vl) if smiles in overlap_train_val]

            # sort for the idx:
            occurences = sorted(occurences, key=lambda x: x[1])
            
            raise ValueError(f"Overlap detected in train, val smiles sets, len(overlap):\n{len(overlap_train_val)}\nOccurences ((dsname, id), idx):\n{occurences}")

        print("No overlap in train, val smiles sets")

        overlap_train_test = list(set(tr).intersection(set(te)))
        if overlap_train_test:
            occurances = [(x, overlap_train_test.index(smiles)) for x, smiles in zip(tr_ids+te_ids, tr+te) if smiles in overlap_train_test]

            # sort for the idx:
            occurances = sorted(occurances, key=lambda x: x[1])

            raise ValueError(f"Overlap detected in train, test smiles sets, len(overlap):\n{len(list(overlap_train_test))}\nOccurences ((dsname, id), idx):\n{occurances}")

        print("No overlap in train, test smiles sets")

        overlap_val_test = list(set(vl).intersection(set(te)))
        if overlap_val_test:
            occurances = [(x, overlap_val_test.index(smiles)) for x, smiles in zip(vl_ids+te_ids, vl+te) if smiles in overlap_val_test]

            # sort for the idx:
            occurances = sorted(occurances, key=lambda x: x[1])

            raise ValueError(f"Overlap detected in val, test smiles sets, len(overlap):\n{len(overlap_val_test)}\nOccurences ((dsname, id), idx):\n{occurances}")
        
        print("No overlap in val, test smiles sets, check passed")


import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dspath", type=str, default='/hits/fast/mbm/seutelf/data/datasets', help="Path to dataset")
    parser.add_argument("--splitpath", type=str, default='/hits/fast/mbm/seutelf/data/splits', help="Path to save split")
    parser.add_argument("--partition", type=float, nargs='+', default=(0.8, 0.1, 0.1), help="Partition of train, val, test")
    parser.add_argument("--split_id", type=int, default=None, help="Split ID")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--check", action='store_true', help="Check split")
    parser.add_argument("--create_duplicates", action='store_true', help="Create duplicates.json file")

    args = parser.parse_args()

    create_split(args.dspath, args.splitpath, args.partition, args.split_id, args.seed, check=args.check, create_duplicates=args.create_duplicates)