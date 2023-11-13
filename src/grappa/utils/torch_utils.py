import torch
import numpy as np
from typing import List, Union, Tuple, Dict
import copy
import random

def calc_split_ids(ids:list[str], ds_names:List[str], partition:Union[Tuple[float,float,float], Dict[str, Tuple[float, float, float]]], seed:int=0, duplicate_partition:Tuple[float,float,float]=(0.8,0.1,0.1)):
    """
    Returns a dictionary containing the molecule ids for train, validation and test sets. The ids are sampled such that smaller datasets also have a share approximate to the given partition.
    partition can be a tuple of floats or a dict of tuples of floats. If it is a tuple, the same partition is used for all datasets. If it is a dict, the keys must be the dataset names and the values must be tuples of floats.
    """

    out = {"train":[], "val":[], "test":[]} # dict of lists of ids

    # indices of those ids that occur more than once
    duplicate_indices = [i for i, x in enumerate(ids) if ids.count(x) > 1]

    unique_indices = [i for i in range(len(ids)) if i not in duplicate_indices]

    duplicates = copy.deepcopy([ids[idx] for idx in duplicate_indices])

    uniques = {ds_name:[] for ds_name in sorted(set(ds_names))}
    for idx in unique_indices:
        dsname = ds_names[idx]
        uniques[dsname].append(ids[idx])

    uniques = copy.deepcopy(uniques)


    # First, split all duplicates. For this, use the partition that belongs to the first dataset in the partition dict that occurs in the duplicates list.

    dupl_partition = duplicate_partition

    # make each id appear once in duplicates without using set to preserve deterministic behaviour
    unique_duplicates = []
    for idx in duplicate_indices:
        if ids[idx] not in unique_duplicates:
            unique_duplicates.append(ids[idx])
    duplicates = unique_duplicates

    random.seed(seed)
    random.shuffle(duplicates)
    total_count = len(duplicates)
    
    n_train = int(total_count * dupl_partition[0])
    n_val = int(total_count * dupl_partition[1])
    n_test = total_count - n_train - n_val

    dup_train = duplicates[:n_train]
    dup_val = duplicates[n_train:n_train+n_val]
    dup_test = duplicates[n_train+n_val:]
    assert len(dup_test) == n_test

    assert len(dup_train) + len(dup_val) + len(dup_test) == total_count, f"Duplicates split failed, {len(dup_train) + len(dup_val) + len(dup_test)} != {total_count}"


    # count how many duplicate ids are in each dataset
    ds_counts = {
        ds_name:{'train':0, 'val':0, 'test':0}
        for ds_name in set(ds_names)
    }
    for idx in duplicate_indices:
        ds_counts[ds_names[idx]]['train'] += 1 if ids[idx] in dup_train else 0
        ds_counts[ds_names[idx]]['val'] += 1 if ids[idx] in dup_val else 0
        ds_counts[ds_names[idx]]['test'] += 1 if ids[idx] in dup_test else 0


    # now split the uniques such that for each ds_name, the ratio of train, val and test is approx the same as in the partition dict
    # Shuffle
    for dsname in uniques.keys():
        if not isinstance(partition, tuple):
            ds_partition = partition[dsname]
        else:
            ds_partition = partition

        assert abs(sum(ds_partition) - 1) <= 1e-10, "Partitions must sum to 1.0"
        assert len(ds_partition) == 3, "Partitions must be a tuple of length 3"

        these_uniques = uniques[dsname]

        random.shuffle(these_uniques)

        # Calculate remaining number of samples needed for each partition to reach partition size
        n_dup_train, n_dup_val, n_dup_test = ds_counts[dsname]['train'], ds_counts[dsname]['val'], ds_counts[dsname]['test']        

        total_count = len(these_uniques) + n_dup_train + n_dup_val + n_dup_test

        n_add_train = max(int(total_count * ds_partition[0]) - n_dup_train, 0)
        n_add_val = max(int(total_count * ds_partition[1]) - n_dup_val, 0)
        n_add_test = len(these_uniques) - n_add_train - n_add_val

        # redistribute if necessary
        while n_add_test < 0:
            # remove from train or val
            if n_add_train > 0:
                n_add_train -= 1
            elif n_add_val > 0:
                n_add_val -= 1
            else:
                raise ValueError("Not enough samples to fill test set")
            n_add_test += 1

        this_train = these_uniques[:n_add_train]
        this_val = these_uniques[n_add_train:n_add_train+n_add_val]
        this_test = these_uniques[n_add_train+n_add_val:]

        assert len(this_train) + len(this_val) + len(this_test) == len(these_uniques), f"Uniques split failed, {len(this_train) + len(this_val) + len(this_test)} != {len(these_uniques)}"

        out['train'] += this_train
        out['val'] += this_val
        out['test'] += this_test

    out['train'] += dup_train
    out['val'] += dup_val
    out['test'] += dup_test

    assert len(out['train']) + len(out['val']) + len(out['test']) == len(set(list(ids))), f"Split failed, {len(out['train']) + len(out['val']) + len(out['test'])} != {len(ids)}"

    assert len(set(out['train']).intersection(set(out['val']))) == 0, "Train and val sets must not overlap"
    assert len(set(out['train']).intersection(set(out['test']))) == 0, "Train and test sets must not overlap"
    assert len(set(out['val']).intersection(set(out['test']))) == 0, "Val and test sets must not overlap"

    return out


def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return torch.sqrt(torch.mean(torch.square(y_true - y_pred)))

def invariant_mae(y_true, y_pred):
    """
    Calculates the mean absolute error between y_true and y_pred, but invariant to rotations assuming that the last dimension of y_true and y_pred are spatial. (i.e. use the compnent rmse as absolute error for each instance.)
    """
    diffs = torch.sqrt(torch.sum(torch.square(y_true - y_pred), dim=-1))
    return torch.mean(diffs)

def invariant_rmse(y_true, y_pred):
    """
    Calculates the root mean squared error between y_true and y_pred, but invariant to rotations assuming that the last dimension of y_true and y_pred are spatial. (i.e. use the compnent rmse as absolute error for each instance.)
    In the case of RMSE, this means: invariant_rmse = sqrt(3) * rmse
    """
    diffs = torch.sum(torch.square(y_true - y_pred), dim=-1)
    return torch.sqrt(torch.mean(diffs))