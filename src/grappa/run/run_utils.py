from pathlib import Path
import numpy as np
import random
import json
from typing import Tuple, List, Union
import dgl

import yaml
import math
import os
import shutil
import copy

from grappa.training.utilities import shape_test


def truncate_string(string, max_len=15):
    if len(string) > max_len:
        if max_len > 9:
            string = string[:max_len-6] + "..." + string[-3:]
        else:
            string = string[:max_len-3] + "..."
    return string


def get_pretrain_name(storage_path,vname, idx=None):
    return get_version_name(idx=idx, storage_path=storage_path, vname=vname) + "_pretrain"


def get_version_name(storage_path,vname='', idx=None):
    """
    Version names are indexed in the order they are created. The creation-index is placed in the beginning. If an index is specified, the version name is appended with this additional index.
    """
    storage_path = Path(storage_path)
    version_indices = [get_version_idx(e) for e in storage_path.glob("*")]
    if len(version_indices) == 0:
        new_idx = 0
    else:
        new_idx = max(version_indices)+1
    if vname == '':
        vname = f"{new_idx}"
    else:
        vname = f"{new_idx}_"+vname

    if not idx is None:
        vname += f"_{idx}"
    return vname


# for each configuration of the model and train algorithm, we create a new version
def get_version_idx(version_path:Path): 
    return int(version_path.stem.split("_")[0])

# for each repetition of the training the same model, we create a new experiment
def get_experiment_idx(version_path:Path):
    if len(version_path.stem.split("_")) == 1:
        return 0
    else:
        return int(version_path.stem.split("_")[1])


def get_ds_path(ds_tag, ds_base):
    ds_path = str(Path(ds_base)/Path(f"{ds_tag}_dgl.bin"))
    return ds_path



def reduce_mols(ds_tr:list, mols, seed=0):
    """
    Modifies the dataset inplace.
    """
    if not mols is None:
        if mols > len(ds_tr):
            mols = len(ds_tr)
        # shuffle train set and take the first mols
        random.Random(seed).shuffle(ds_tr)
        ds_tr[:] = ds_tr[:mols]


def reduce_confs(ds_tr:list, confs, seed=0):
    """
    Modifies the dataset inplace.
    """
    if confs is None:
        return
    def reduce_confs_(g):
        n_orig = g.nodes["n1"].data["xyz"].shape[1]
        if n_orig > confs:
            np.random.seed(seed)
            indices = np.random.choice(np.arange(n_orig), size=confs, replace=False)
            for key in g.nodes["g"].data.keys():
                if "u_" in key:
                    g.nodes["g"].data[key] = g.nodes["g"].data[key][:,indices]
            for key in g.nodes["n1"].data.keys():
                if "xyz" in key or "grad" in key:
                    g.nodes["n1"].data[key] = g.nodes["n1"].data[key][:,indices]
        return g
    ds_tr[:] = [reduce_confs_(g) for g in ds_tr]


def get_splits(datasets:list, datanames:list, seed:int=0, fractions:Tuple[float, float, float]=[0.8,0.1,0.1], splits:list=None):
    """
    returns 3 datasets and a list of 3 lists containing sequence names used in the split in the order train, val, test
    make sure that no same molecules are in different splits by using the names.
    collect all names, shuffle them and split them into train, val, test
    then assign the graphs to the splits according to the names
    this can result in different fractions for the total number of mols/confs than the ones given
    if the splits argument is specified, creates a new split only containing names that are not in the split already according to the fractions and adds the new split to the existing splits.
    """

    loaded_split = True
    if splits is None:
        splits = [[],[],[]]
        loaded_split = False

    # split the names not occuring in splits according to fraction:
    all_names = []
    for names in datanames:
        for name in names:
            if not name in all_names and not any(name in split for split in splits):
                all_names.append(name)

    if not loaded_split:
        assert len(all_names)>3, "not enough molecules for splitting"

    random.Random(seed).shuffle(all_names)
    
    # hard code the case of less than 4 molecules:
    if len(all_names) == 4:
        nums = [2,1,1]
    elif len(all_names) == 3:
        nums = [1,1,1]
    elif len(all_names) == 2:
        nums = [1,1,0]
    elif len(all_names) == 1:
        nums = [all_names[0]]*3
        print("Warning: only one molecule in dataset, using it for train, val and test to test overfitting capabilities.")
    else:
        nums = [math.ceil(len(all_names)*f/sum(fractions)) for f in fractions]

        # make sure that the sum of the numbers is the total number of names
        nums[0] += len(all_names) - sum(nums)


    new_splits = [all_names[:nums[0]]]
    s = nums[0]
    for i in range(1,len(nums)):
        new_splits.append(all_names[s:s+nums[i]])
        s += nums[i]

    if loaded_split:
        print(f"loaded split: train {len(splits[0])}, val {len(splits[1])}, test {len(splits[2])}")
        print(f"add. split  : train {len(new_splits[0])}, val {len(new_splits[1])}, test {len(new_splits[2])}\n")

    for i in range(len(splits)):
        splits[i].extend(new_splits[i])
            
    print(f"splitted molecule names: train {len(splits[0])}, val {len(splits[1])}, test {len(splits[2])}\n")

    # fill the dataset lists according to the names:
    ds_trs, ds_vls, ds_tes = [], [], []
    for i in range(len(datasets)):
        ds_tr, ds_vl, ds_te = [], [], []
        for j in range(len(datasets[i])):
            if datanames[i][j] in splits[0]:
                ds_tr.append(datasets[i][j])
            elif datanames[i][j] in splits[1]:
                ds_vl.append(datasets[i][j])
            elif datanames[i][j] in splits[2]:
                ds_te.append(datasets[i][j])
            else:
                raise ValueError("name not in any split")
        ds_trs.append(ds_tr)
        ds_vls.append(ds_vl)
        ds_tes.append(ds_te)

    return ds_trs, ds_vls, ds_tes, splits


def flatten_splits(ds_trs, ds_vls, ds_tes):
    ds_tr = []
    ds_vl = []
    ds_te = []
    for i in range(len(ds_trs)):
        ds_tr.extend(ds_trs[i])
        ds_vl.extend(ds_vls[i])
        ds_te.extend(ds_tes[i])

    print(f"splitted set of molecular graphs: train {len(ds_tr)}, val {len(ds_vl)}, test {len(ds_te)}\n")
    return ds_tr, ds_vl, ds_te

def get_data(ds_paths:List[Union[str, Path]], n_graphs=None, force_factor=0):
    datasets = []
    datanames = []
    for p in ds_paths:
        ############
        idxs = None
        if not n_graphs is None:
            idxs = [e for e in range(n_graphs)]
            print(f"Using {n_graphs} graphs")
        ############

        print(f"loading dataset {p}...\n")

        assert Path(p).exists(), f"dataset {p} does not exist"

        try:
            ds, _ = dgl.load_graphs(str(p), idx_list=idxs)
        except:
            if n_graphs is None:
                raise
            else:
                # assume that the error is due to n_graphs being smaller than the number of graphs in the dataset
                ds, _ = dgl.load_graphs(p)
                if len(ds) >=n_graphs:
                    raise
        
        try:
            seqpath = str(Path(p).parent/Path(p).stem) + "_seq.json"
            with open(seqpath, "r") as f:
                names = json.load(f)
            if not n_graphs is None:
                names = names[:n_graphs]
            if len(names) != len(ds):
                raise ValueError(f"number of names does not match number of graphs: {len(names)} != {len(ds)}")
            print(f"loaded {len(ds)} graphs and their names\n")
        except Exception as e:
            names = None
            print(f"loaded {len(ds)} graphs. couldnt load their names due to {e}\n")

        ############
        shape_test(ds, force_factor=force_factor)
        ############
        datasets.append(ds)
        datanames.append(names)

    if len(datasets) > 1:
        for i in range(len(datasets)):
            assert len(datasets[i]) == len(datanames[i])

    return datasets, datanames


def load_yaml(path:Union[str,Path]):
    with open(str(path), 'r') as f:
        d = yaml.safe_load(f)
    return d

def store_yaml(d:dict, path:Union[str,Path]):
    with open(str(path), 'w') as f:
        yaml.dump(d, f)


# def clean_vpath(storage_path, must_contain="log.txt"):
#     for e in Path(storage_path).glob("*"):
#         if not Path(e)/Path(must_contain) in e.glob("*"):
#             shutil.rmtree(e)
            

def init_dirs(storage_path:str, idx:int, vname:str='', continue_path:str=None):

    storagepath = storage_path
    if not continue_path is None:
        # version name is the last part of continue_path, storage is the rest

        version_name = Path(continue_path).name
        pretrain_name = version_name + "_pretrain"
        storagepath = str(Path(continue_path).parent)
    else:
        version_name = get_version_name(idx=idx, storage_path=storagepath, vname=vname)
        pretrain_name = get_pretrain_name(idx=idx, storage_path=storagepath, vname='')

    vpath = os.path.join(storagepath,version_name)
    ppath = os.path.join(vpath,pretrain_name)
    os.makedirs(vpath, exist_ok=True)
    os.makedirs(ppath, exist_ok=True)

    return str(version_name), str(pretrain_name)


def write_config(args, idx=None):
    version_name, pretrain_name = init_dirs(storage_path=args["storage_path"], idx=idx, vname=args["name"], continue_path=args["continue_path"])
    args["version_name"] = version_name
    args["pretrain_name"] = pretrain_name
    args_ = copy.deepcopy(args)
    args_.pop("load_path")
    args_.pop("continue_path")

    conf_p = Path(args["storage_path"])/Path(version_name)/Path("config.yaml")

    if not args["continue_path"] is None:
        for file in conf_p.parent.glob("*.yaml"):
            # rename the old config:
            if len(file.name.split("-")) != 1:
                k = int(file.name.split("-")[0])
                new_name = str(k+1) + "-" + "".join(file.name.split("-")[1:])
            else:
                new_name = "1-" + file.name
            shutil.copy(str(file), os.path.join(str(file.parent), new_name))

    store_yaml(args_, str(conf_p))


def get_loaders(datasets:Tuple[List[dgl.graph]])->Tuple[dgl.dataloading.GraphDataLoader]:
    return tuple([dgl.dataloading.GraphDataLoader(ds) for ds in datasets])


def get_all_loaders(subsets, ds_paths, ds_tags=None, basename="te"):
    """
    returns a list of loaders for the whole dataset and for each subset. the loader of the whole dataset is the zeroth element of the list.
    """
    assert len(subsets) == len(ds_paths)

    # the whole dataset:
    te_loaders = [dgl.dataloading.GraphDataLoader([g for subset in subsets for g in subset])]

    te_names = [basename]

    if len(subsets) > 1:
        te_loaders += list(get_loaders((subset for subset in subsets)))
        if not ds_tags is None:
            assert len(ds_tags) == len(subsets), f"number of tags {len(ds_tags)} and datasets {len(subsets)} do not match. they have to be either None or be of the same count as the subsets"
            te_subnames = [basename+"_"+t for t in ds_tags]
        else:
            te_subnames = [f"{basename}_{i}: {ds_paths[i]}" for i in range(len(subsets))]
        te_names += te_subnames

    if len(te_loaders) != len(te_names):
        raise ValueError(f"number of loaders ({len(te_loaders)}) and names ({len(te_names)}) do not match, test names:\n{te_names}")
    
    return te_loaders, te_names
