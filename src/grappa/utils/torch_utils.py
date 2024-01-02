import torch
import numpy as np
from typing import List, Union, Tuple, Dict
import copy
import random

from typing import List, Tuple, Union, Dict
import copy, random, numpy as np
from collections import defaultdict


def do_fold(ids:List, k:int, num_folds:int=None):
    """
    Helper function. Returns a tuple of k lists of ids: tr, vl, te.
    The lists satisfy the following properties:
    - len(te) ~ len(ids)/k
    - len(vl) ~ len(te)
    - len(tr[i]) = len(ids[i]) - len(te[i]) - len(vl[i])
    - tr[i], vl[i], te[i] are pairwise disjoint

    - The set of test splits is a disjoint partition of the ids, i.e.:
        concat(te[i] for i in range(k)) = ids
    """

    assert k > 2, "k must be larger than 2"
    if num_folds is None:
        num_folds = k

    # partition = [(k-2.)/k, 1./k, 1./k]

    
    train_id_list = []
    val_id_list = []
    test_id_list = []

    n = len(ids)

    for i in range(num_folds):
        start_test = int(i*len(ids)/k)
        end_test = int((i+1)*len(ids)/k)

        start_val = end_test
        end_val = int((i+2)*len(ids)/k)

        test_ids = ids[start_test:end_test]
        if end_val < n:
            val_ids = ids[start_val:end_val]
            train_ids = ids[:start_test] + ids[end_val:]
        else: 
            end_val = end_val - n
            # now end_val < start_val and start val is on the very right side of the list
            val_ids = ids[:end_val] + ids[start_val:]
            train_ids = ids[end_val:start_test]

        train_id_list.append(train_ids)
        val_id_list.append(val_ids)
        test_id_list.append(test_ids)

    return train_id_list, val_id_list, test_id_list


def get_k_fold_split_ids(ids:list[str], ds_names:List[str], k:int=10, seed:int=0, num_folds:int=None):
    """
    Returns a list of dictionaries containing the molecule ids for train, validation and test sets. The ids are sampled such that smaller datasets also have a share approximate to the given partition.
    The ids are split into k_fold folds. The ids are split such that all test sets concatenated are the full set of ids. The train and val sets are split such that the val set is approximately the same size as the test set. E.g. for n=10, we split according to (0.8,0.1,0.1) 10 times.
    """

    random.seed(seed)
    np.random.seed(seed)

    if num_folds is None:
        num_folds = k


    # indices and dsnames of those ids that occur more than once
    duplicates = [(i,dsname) for i, (x, dsname) in enumerate(zip(ids, ds_names)) if ids.count(x) > 1]

    duplicate_indices = set([idx for idx, _ in duplicates]) # set for faster lookup

    unique_indices = [i for i in range(len(ids)) if i not in duplicate_indices]

    # dictionary containing the unique ids for each dataset name
    uniques = {ds_name:[] for ds_name in sorted(set(ds_names))}
    for idx in unique_indices:
        dsname = ds_names[idx]
        uniques[dsname].append(ids[idx])

    # dictionary containing the dsnames of each duplicate id
    duplicate_dsnames = defaultdict(list)
    for idx, dsname in duplicates:
        duplicate_dsnames[ids[idx]].append(dsname)

    # now distribute the duplicates by picking a random subdataset for each duplicate id
    for id, dsnames in duplicate_dsnames.items():
        dsname = random.choice(dsnames)
        uniques[dsname].append(id)

    uniques = copy.deepcopy(uniques)

    # now we can do a partition for each dataset individually since we know that all ids only occur once, also along different datasets

    out = [{'train':[], 'val':[], 'test':[]} for _ in range(num_folds)]

    for dsname in uniques.keys():

        these_uniques = uniques[dsname]

        random.shuffle(these_uniques)

        this_train_split, this_val_split, this_test_split = do_fold(these_uniques, k, num_folds=num_folds)

        for i in range(num_folds):
            out[i]['train'] += this_train_split[i]
            out[i]['val'] += this_val_split[i]
            out[i]['test'] += this_test_split[i]

    # done, now check whether the splits are correct:

    # first, check whether each fold has disjoint tr, vl, te:
    for i in range(num_folds):
        
        assert len(out[i]['train']) + len(out[i]['val']) + len(out[i]['test']) == len(set(list(ids))), f"Split failed, {len(out[i]['train']) + len(out[i]['val']) + len(out[i]['test'])} != {len(ids)}"

        assert len(set(out[i]['train']).intersection(set(out[i]['val']))) == 0, "Train and val sets must not overlap"
        assert len(set(out[i]['train']).intersection(set(out[i]['test']))) == 0, "Train and test sets must not overlap"
        assert len(set(out[i]['val']).intersection(set(out[i]['test']))) == 0, "Val and test sets must not overlap"

    if num_folds == k:
        # verify that the set of test splits is a disjoint partition of the ids:
        test_ids = []
        for i in range(num_folds):
            test_ids += out[i]['test']

        assert len(set(test_ids)) == len(test_ids), "Test ids must be unique"
        assert set(test_ids) == set(ids), "Test ids must be a partition of the full set of ids"
        

    return out


def calc_split_ids(ids:list[str], ds_names:List[str], partition:Union[Tuple[float,float,float], Tuple[Tuple[float,float,float],Dict[str, Tuple[float, float, float]]]], seed:int=0, duplicate_partition:Tuple[float,float,float]=(0.8,0.1,0.1)):
    """
    Returns a dictionary containing the molecule ids for train, validation and test sets. The ids are sampled such that smaller datasets also have a share approximate to the given partition.
    partition can be a tuple of three floats or a tuple of a (default) tuple of 3 floats and a dict mapping dsnames to tuples of 3 floats. If it is a tuple, the same partition is used for all datasets. If it is a dict, the keys must be the dataset names and the values must be tuples of floats.
    It can be guaranteed that if one dsname partition is such that all molecules are in either train, val or test, then the same is true for molecules with the same id in other datasets.
    """
    random.seed(seed)

    def get_partition_tuple(dsname):
        out = None
        if isinstance(partition[1], dict):
            if dsname in partition[1].keys():
                out = tuple(partition[1][dsname])
            else:
                out = tuple(partition[0])
        elif isinstance(partition, tuple):
            out = partition
        elif isinstance(partition, list):
            out = tuple(partition)
        else:
            raise ValueError(f"Unknown type for partition: {type(partition)}")
        if not all([x>=0. for x in out]):
            raise ValueError(f"Partition tuple for {dsname} contains negative values: {out}")
        if abs(sum(list(out))-1.) > 1e-10:
            raise ValueError(f"Partition tuple for {dsname} does not sum to 1.0: {out}")
        return out

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
    unique_duplicate_ids = []
    for idx in duplicate_indices:
        if ids[idx] not in unique_duplicate_ids:
            unique_duplicate_ids.append(ids[idx])
    duplicates = unique_duplicate_ids


    # assign those ids that have to be in the train, val or test set because they are in a dataset that has a partition of (1,0,0), (0,1,0) or (0,0,1):
    #############################
    # now loop again over the duplicates, this time also considering the ds_names. If the individual partition of a dsname has a one at some place, we can remove the id from the duplicates list and directly assign it to the corresponding set since it must be there then.
    # first, create a dict of duplicate_id:dsnames:
    duplicate_id_dsnames = {id:[] for id in duplicates}
    for idx in duplicate_indices:
        duplicate_id_dsnames[ids[idx]].append(ds_names[idx])
    
    # now loop over the duplicates, assert that there is no one at different positions for different ds_names and assign to the corresponding set
    for id in duplicates:
        has_to_be_in = None
        for dsname in duplicate_id_dsnames[id]:
            partition_tuple = get_partition_tuple(dsname)
            if any([abs(x-1.) < 1e-10 for x in partition_tuple]):
                # find the index of the one (we can simply take the max):
                idx = int(np.argmax(partition_tuple))
                if not has_to_be_in is None:
                    if has_to_be_in != idx:
                        raise ValueError(f"Internal error: Duplicate id {id} has to be in both {has_to_be_in} and {idx}.")
                else:
                    has_to_be_in = idx
        
        if not has_to_be_in is None:
            out['train'] += [id] if has_to_be_in == 0 else []
            out['val'] += [id] if has_to_be_in == 1 else []
            out['test'] += [id] if has_to_be_in == 2 else []

    # remove all that have been assigned
    for id in out['train'] + out['val'] + out['test']:
        duplicates.remove(id)
        duplicate_id_dsnames.pop(id)

    # the dataset names of the remaining duplicates:
    duplicate_names = []
    for id in duplicates:
        duplicate_names += duplicate_id_dsnames[id]
    duplicate_names = set(duplicate_names)            

    # check whether there can be problems with the partitioning of duplicates after removing those ids that have to be in one of the sets
    if isinstance(partition[1], dict):
        # assert that all dsnames in duplicate_names have the duplicate partition:
        for dsname in duplicate_names:
            if dsname in partition[1].keys():
                assert partition[1][dsname] == duplicate_partition, f"Partition for {dsname} ({partition[1][dsname]}) does not match duplicate partition ({duplicate_partition}) although dataset has duplicate ids."
    
    #############################

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
        if isinstance(partition[1], dict):
            if dsname in partition[1].keys():
                ds_partition = partition[1][dsname]
            else:
                ds_partition = partition[0]
        elif isinstance(partition, tuple):
            ds_partition = partition
        elif isinstance(partition, list):
            ds_partition = tuple(partition)
        else:
            raise ValueError(f"Unknown type for partition: {type(partition)}")

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
    if y_true.shape[-1] != 3:
        raise ValueError("y_true must have shape (..., 3) for invariant_mae")
    diffs = torch.sqrt(torch.sum(torch.square(y_true - y_pred), dim=-1))
    return torch.mean(diffs)

def invariant_rmse(y_true, y_pred):
    """
    Calculates the root mean squared error between y_true and y_pred, but invariant to rotations assuming that the last dimension of y_true and y_pred are spatial. (i.e. use the compnent rmse as absolute error for each instance.)
    In the case of RMSE, this means: invariant_rmse = sqrt(3) * rmse
    """
    if y_true.shape[-1] != 3:
        raise ValueError("y_true must have shape (..., 3) for invariant_rmse")
    diffs = torch.sum(torch.square(y_true - y_pred), dim=-1)
    return torch.sqrt(torch.mean(diffs))