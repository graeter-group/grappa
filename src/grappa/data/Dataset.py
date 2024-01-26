"""
Defines a Dataset class that handles dataset splitting and batching. Enables creation of a dgl dataset from a set of MolData objects.
Datasets can be saved and loaded from disk.

If initialized with a list of MolData objects, stores a list the mol_ids and a list of dgl graphs in corresponding order. These are used to create splits into train, validation and test sets in which no molecule is present in more than one set. The splits are stored as lists of indices.
"""

from grappa.data import MolData
from grappa.utils import torch_utils, dataset_utils
import dgl
from dgl import DGLGraph
import json
from pathlib import Path

from typing import List, Union, Tuple, Dict

import numpy as np

import torch

# inherit from torch ds:
class Dataset(torch.utils.data.Dataset):
    """
    Class that stores dgl graphs, their mol_ids and the name of the subdataset to which they belong.
    Items are returned as (graph, subdataset) tuples.
    The mol_ids are used to create splits into train, validation and test sets.
    """
    def __init__(self, graphs:List[DGLGraph]=[], mol_ids:List[str]=[], subdataset:Union[List[str], str]=[]):
        """
        Args:
            graphs (List[DGLGraph]): list of dgl graphs
            mol_ids (List[str]): list of molecule ids
            subdataset (List[str]): list of subdataset names
        """
        if isinstance(subdataset, str):
            subdataset = [subdataset]*len(graphs)
        self.graphs = graphs
        self.mol_ids = mol_ids
        self.subdataset = subdataset
        assert len(graphs) == len(mol_ids) == len(subdataset)

    def from_tag(tag:str, data_dir:Union[str, Path]=dataset_utils.get_data_path()/'dgl_datasets'):
        """
        Returns a Dataset object from a tag. Downloads the dataset if it is not already present in the data_dir.
        Args:
            tag (str): tag of the dataset to be loaded
            data_dir (Union[str, Path]): path to the directory where the dataset is stored. Default: 'grappa/data/dgl_datasets'
        Returns:
            dataset (Dataset): Dataset object containing the graphs, mol_ids and subdataset names

        Possible tags are:
        BENCHMARK ESPALOMA:
            - 'spice-dipeptide'
            - 'spice-des-monomers'
            - 'spice-pubchem'
            ...
        PEPTIDE DATASET:
            - 'tripeptide_amber99sbildn'
            - 'spice_dipeptide_amber99sbildn'
        """
        path = dataset_utils.get_path_from_tag(tag=tag, data_dir=data_dir)
        return Dataset.load(path)


    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.subdataset[idx]
    
    def split(self, train_ids:List[str], val_ids:List[str], test_ids:List[str], check_overlap:bool=True):
        """
        Splits the dataset into train, validation and test sets that are then returned as new Dataset objects.
        Args:
            train_ids (List[str]): list of molecule ids to be used for training
            val_ids (List[str]): list of molecule ids to be used for validation
            test_ids (List[str]): list of molecule ids to be used for testing
        Returns:
            train, val, test (Dataset): Dataset objects containing the respective splits
        """
        train_ids = set(train_ids)
        val_ids = set(val_ids)
        test_ids = set(test_ids)
        if check_overlap:
            assert len(train_ids.intersection(val_ids)) == 0
            assert len(train_ids.intersection(test_ids)) == 0
            assert len(val_ids.intersection(test_ids)) == 0

        assert all([mol_id in val_ids.union(train_ids).union(test_ids) for mol_id in self.mol_ids]), "Some molecule ids are not in the split"

        train_idx = [i for i in range(len(self.mol_ids)) if self.mol_ids[i] in train_ids]
        val_idx = [i for i in range(len(self.mol_ids)) if self.mol_ids[i] in val_ids]
        test_idx = [i for i in range(len(self.mol_ids)) if self.mol_ids[i] in test_ids]

        train_graphs, train_mol_ids, train_subdataset = map(list, zip(*[(self.graphs[i], self.mol_ids[i], self.subdataset[i]) for i in train_idx]))
        val_graphs, val_mol_ids, val_subdataset = map(list, zip(*[(self.graphs[i], self.mol_ids[i], self.subdataset[i]) for i in val_idx]))
        test_graphs, test_mol_ids, test_subdataset = map(list, zip(*[(self.graphs[i], self.mol_ids[i], self.subdataset[i]) for i in test_idx]))

        train = Dataset(graphs=train_graphs, mol_ids=train_mol_ids, subdataset=train_subdataset)
        val = Dataset(graphs=val_graphs, mol_ids=val_mol_ids, subdataset=val_subdataset)
        test = Dataset(graphs=test_graphs, mol_ids=test_mol_ids, subdataset=test_subdataset)
        
        return train, val, test
    

    def save(self, path:Union[str, Path]):
        """
        Saves the dataset to disk at the given directoy. Saves the graphs at graphs.bin via dgl and mol_ids, subdataset as json lists.
        Args:
            path (Union[str, Path]): path to a directory where the dataset should be saved
        """

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        dgl.save_graphs(str(path / 'graphs.bin'), self.graphs)
        with open(path / 'mol_ids.json', 'w') as f:
            json.dump(self.mol_ids, f)
        with open(path / 'subdataset.json', 'w') as f:
            json.dump(self.subdataset, f)
        

    @classmethod
    def load(cls, path:Union[str, Path]):
        """
        Loads a dataset from disk. Loads the graphs from graphs.bin via dgl and mol_ids, subdataset as json lists.
        Args:
            path (Union[str, Path]): path to a directory where the dataset is saved
        Returns:
            dataset (Dataset): Dataset object containing the graphs, mol_ids and subdataset names
        """
        path = Path(path)
        graphs, _ = dgl.load_graphs(str(path / 'graphs.bin'))
        with open(path / 'mol_ids.json', 'r') as f:
            mol_ids = json.load(f)
        with open(path / 'subdataset.json', 'r') as f:
            subdataset = json.load(f)
        return cls(graphs, mol_ids, subdataset)
    

    @classmethod
    def from_moldata(cls, moldata_list:List[MolData], subdataset:List[str]):
        """
        Creates a Dataset object from a list of MolData objects.
        Args:
            moldata_list (List[MolData]): list of MolData objects
            subdataset (List[str]): list of subdataset names
        Returns:
            dataset (Dataset): Dataset object containing the graphs, mol_ids and subdataset names
        """
        graphs = [moldata.to_dgl() for moldata in moldata_list]
        mol_ids = [moldata.mol_id for moldata in moldata_list]
        return cls(graphs, mol_ids, subdataset)
    

    def __add__(self, other):
        """
        Concatenates two datasets.
        Args:
            other (Dataset): dataset to be concatenated
        Returns:
            dataset (Dataset): concatenated dataset
        """
        graphs = self.graphs + other.graphs
        mol_ids = self.mol_ids + other.mol_ids
        subdataset = self.subdataset + other.subdataset
        return Dataset(graphs, mol_ids, subdataset)


    def remove_uncommon_features(self, create_feats:Dict[str, Union[float,torch.Tensor]]={'is_radical':0.}):
        """
        Removes features that are not present in all graphs. This is necessary for batching.
        ----------
        Parameters
        ----------
        create_feats : Dict[str, torch.Tensor], optional
            Dictionary of features that should be created for all graphs if not already present in the graph. The key is the name of the feature and the value is the default value per node for the feature. For example, if some graphs have the is_radical feature as onedimensional onehot encoding, create_feats={'is_radical':0} will make all molecules without this feature have torch.zeros((n_atoms, 1)) as is_radical feature.
        """
        for v in create_feats.values():
            if isinstance(v, torch.Tensor):  
                v.to(self.graphs[0].nodes['n1'].data['xyz'].device)

        def add_feats(graph):
            for k, v in create_feats.items():
                if not k in graph.nodes['n1'].data.keys():
                    if isinstance(v, torch.Tensor):
                        graph.nodes['n1'].data[k] = torch.repeat_interleave(v, graph.num_nodes('n1'), dim=0)
                    else:
                        graph.nodes['n1'].data[k] = torch.ones(graph.num_nodes('n1'))*v
            return graph
        
        # add feats to first graph so that they are included in keep
        self.graphs[0] = add_feats(self.graphs[0])

        removed = set()
        keep = set(self.graphs[0].ndata.keys())

        # iterate twice, first to collect all feats that are to be kept, then to remove all feats that are not to be kept
        for i in range(len(self.graphs)):
            self.graphs[i] = add_feats(self.graphs[i])
            keep = keep.intersection(set(self.graphs[i].ndata.keys()))

        for graph in self.graphs:
            removed = removed.union(set(graph.ndata.keys()).difference(keep))
            for feature in set(graph.ndata.keys()).difference(keep):
                del graph.ndata[feature]

        if len(removed) > 0:
            print(f"Removed features:\n  {removed}")
    

    def clean(self, keep_feats:List[str]=None):
        """
        Assumes that all graphs have the same node features, i.e. that Dataset.remove_uncommon_features() has been called before.
        Remove all n1 feats with name that is not in keep_feats. This is for speeding up transfer of data to gpu.
        """
        if keep_feats is None:
            return
        remove = []
        graph = self.graphs[0]
        for feature in graph.nodes['n1'].data.keys():
            if not feature in keep_feats + ['xyz', 'atomic_number', 'partial_charge', 'ring_encoding']:
                remove.append(feature)

        for graph in self.graphs:
            for feature in remove:
                del graph.nodes['n1'].data[feature]


    def get_k_fold_split_ids(self, k:int, seed:int=0, num_folds:int=None):
        """
        Returns a list of dictionaries containing the molecule ids for train, validation and test sets. The ids are sampled such that smaller datasets also have a share approximate to the given partition.
        The ids are split into k_fold folds. The ids are split such that all test sets concatenated are the full set of ids. The train and val sets are split such that the val set is approximately the same size as the test set. E.g. for n=10, we split according to (0.8,0.1,0.1) 10 times.
        """
        return torch_utils.get_k_fold_split_ids(ids=self.mol_ids, ds_names=self.subdataset, k=k, seed=seed, num_folds=num_folds)



    def calc_split_ids(self, partition:Union[Tuple[float,float,float], Dict[str, Tuple[float, float, float]]], seed:int=0, existing_split:Dict[str, List[str]]=None):
        """
        Returns a dictionary containing the molecule ids for train, validation and test sets. The ids are sampled such that all (also smaller) datasets have a share approximate to the given partition.
        partition can be a tuple of floats or a dict of tuples of floats. If it is a tuple, the same partition is used for all datasets. If it is a dict, the keys must be the dataset names and the values must be tuples of floats.
        """
        return torch_utils.calc_split_ids(ids=self.mol_ids, partition=partition, seed=seed, ds_names=self.subdataset, existing_split=existing_split)
    

    def slice(self, start, stop):
        # Create a new dataset instance with only the sliced data
        sliced_graphs = self.graphs[start:stop]
        sliced_mol_ids = self.mol_ids[start:stop]
        sliced_subdataset = self.subdataset[start:stop]

        return Dataset(graphs=sliced_graphs, mol_ids=sliced_mol_ids, subdataset=sliced_subdataset)
    
    def where(self, condition=List[bool]):
        """
        Returns a new dataset instance with only the data at those indices where the condition is True.
        """
        sliced_graphs = [self.graphs[i] for i in range(len(self.graphs)) if condition[i]]
        sliced_mol_ids = [self.mol_ids[i] for i in range(len(self.mol_ids)) if condition[i]]
        sliced_subdataset = [self.subdataset[i] for i in range(len(self.subdataset)) if condition[i]]

        return Dataset(graphs=sliced_graphs, mol_ids=sliced_mol_ids, subdataset=sliced_subdataset)
    

    def shuffle(self, seed:int=0):
        """
        Shuffle the dataset in place and return it
        """
        np.random.seed(seed)
        perm = np.random.permutation(len(self.graphs))
        self.graphs[:] = [self.graphs[i] for i in perm]
        self.mol_ids[:] = [self.mol_ids[i] for i in perm]
        self.subdataset[:] = [self.subdataset[i] for i in perm]
        return self