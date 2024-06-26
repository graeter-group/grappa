"""
Defines a Dataset class that handles dataset splitting and batching. Enables creation of a dgl dataset from a set of MolData objects.
Datasets can be saved and loaded from disk.

If initialized with a list of MolData objects, stores a list the mol_ids and a list of dgl graphs in corresponding order. These are used to create splits into train, validation and test sets in which no molecule is present in more than one set. The splits are stored as lists of indices.
"""

from grappa.data.mol_data import MolData
from grappa.utils import data_utils, torch_utils
from grappa import constants
import dgl
from dgl import DGLGraph
import json
from pathlib import Path

from typing import List, Union, Tuple, Dict

import copy
import numpy as np
import logging
import torch
from tqdm import tqdm

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
        self.num_mols = None # keep track of original number of molecules if self.apply_max_confs is called

    @classmethod
    def from_tag(cls, tag:str):
        """
        Returns a Dataset object from a tag.
        If the dataset if it is not already present in the data_dir as graphs.bin, mol_ids.json and subdataset.json in data_dir/tag/, it is constructed from the MolData objects in grappa/data/datasets/tag/. If there are no MolData objects and the tag is in the list of downloadable datasets, the dataset is downloaded from the grappa repository.
        Args:
            tag (str): tag of the dataset to be loaded
            data_dir (Union[str, Path]): path to the directory where the dataset is stored. Default: 'grappa/data/dgl_datasets'
        Returns:
            dataset (Dataset): Dataset object containing the graphs, mol_ids and subdataset names

        Possible tags for download are:
        BENCHMARK ESPALOMA:
            - 'spice-des-monomers'
            - 'spice-pubchem'
            - 'gen2'
            - 'gen2-torsion'
            - 'spice-dipeptide'
            - 'protein-torsion'
            - 'pepconf-dlc'
            - 'rna-diverse'
            - 'rna-trinucleotide'

        PEPTIDE DATASET:
            - dipeptides-300K-openff-1.2.0
            - dipeptides-300K-amber99
            - dipeptides-300K-charmm36
            - dipeptides-1000K-openff-1.2.0
            - dipeptides-1000K-amber99
            - dipeptides-1000K-charmm36
            - uncapped-300K-openff-1.2.0
            - uncapped-300K-amber99
            - dipeptides-hyp-dop-300K-amber99

        RADICAL DATASET:
            - dipeptides-radical-300K
            - bondbreak-radical-peptides-300K
        """

        data_dir=data_utils.get_data_path()/'dgl_datasets'

        dir_path = Path(data_dir) / tag

        # load the dataset directly if it has been created from a moldata path (to ensure that old datasets are overwritten with ones from the new data pipeline)
        if dir_path.exists() and (data_utils.get_data_path()/"datasets"/tag).exists():
            if not ((dir_path/'graphs.bin').exists() and (dir_path/'mol_ids.json').exists() and (dir_path/'subdataset.json').exists()):
                raise ValueError(f'Found directory {dir_path} but not all necessary files: graphs.bin, mol_ids.json, subdataset.json.')
            return Dataset.load(dir_path)
        else:
            # else, construct the dgl dataset from a folder with moldata files, thus, return a moldata path
            moldata_path = data_utils.get_moldata_path(tag)

            self = Dataset.load(moldata_path)
            return self

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.subdataset[idx]
    
    @classmethod
    def concatenate(cls, *others):
        """
        Concatenates the dataset with other datasets.
        Args:
            others (List[Dataset]): datasets to be concatenated
        Returns:
            dataset (Dataset): concatenated dataset
        """
        graphs = []
        mol_ids = []
        subdataset = []
        for dataset in list(others):
            graphs += dataset.graphs
            mol_ids += dataset.mol_ids
            subdataset += dataset.subdataset
        return cls(graphs, mol_ids, subdataset)
    
    def split(self, train_ids:List[str], val_ids:List[str], test_ids:List[str], check_overlap:bool=True):
        """
        Splits the dataset into train, validation and test sets that are then returned as new Dataset objects.
        If an id of the dataset does not appear in any of the splits, it is included in the test set.
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

        train_idx = [i for i in range(len(self.mol_ids)) if self.mol_ids[i] in train_ids]
        val_idx = [i for i in range(len(self.mol_ids)) if self.mol_ids[i] in val_ids]
        # all the rest is test:
        test_idx = [i for i in range(len(self.mol_ids)) if not i in train_idx and not i in val_idx]

        train_graphs, train_mol_ids, train_subdataset = map(list, zip(*[(self.graphs[i], self.mol_ids[i], self.subdataset[i]) for i in train_idx])) if len(train_idx) > 0 else ([], [], [])
        val_graphs, val_mol_ids, val_subdataset = map(list, zip(*[(self.graphs[i], self.mol_ids[i], self.subdataset[i]) for i in val_idx])) if len(val_idx) > 0 else ([], [], [])
        test_graphs, test_mol_ids, test_subdataset = map(list, zip(*[(self.graphs[i], self.mol_ids[i], self.subdataset[i]) for i in test_idx])) if len(test_idx) > 0 else ([], [], [])

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
        if not ((path/'graphs.bin').exists() and (path/'mol_ids.json').exists() and (path/'subdataset.json').exists()):
            # if there are .npz files, assume that this is a moldata path and load the dataset from there:
            paths = list(path.glob('*.npz'))
            if len(paths) > 0:
                logging.info(f'\nProcessing Dataset from {path}:')
                # create the dgl dataset from moldata objects:
                moldata = []
                for molfile in tqdm(paths, desc='Loading .npz files'):
                    moldata.append(MolData.load(str(molfile)))
                self = Dataset.from_moldata(moldata, subdataset=path.name)
                
                dgl_dir_path = data_dir=data_utils.get_data_path()/'dgl_datasets'/path.name
                logging.info(f"\nSaving dgl dataset to {dgl_dir_path}\n")
                self.save(dgl_dir_path)
                return self


        graphs, _ = dgl.load_graphs(str(path / 'graphs.bin'))
        with open(path / 'mol_ids.json', 'r') as f:
            mol_ids = json.load(f)
        with open(path / 'subdataset.json', 'r') as f:
            subdataset = json.load(f)
        return cls(graphs, mol_ids, subdataset)
    

    @classmethod
    def from_moldata(cls, moldata_list:List[MolData], subdataset:Union[str, List[str]]):
        """
        Creates a Dataset object from a list of MolData objects.
        Args:
            moldata_list (List[MolData]): list of MolData objects
            subdataset (List[str]): list of subdataset names
        Returns:
            dataset (Dataset): Dataset object containing the graphs, mol_ids and subdataset names
        """

        if isinstance(subdataset, str):
            subdataset = [subdataset]*len(moldata_list)

        graphs = []
        mol_ids = []

        for moldata in tqdm(moldata_list, desc='Creating graphs'):
            graphs.append(moldata.to_dgl())
            mol_ids.append(moldata.mol_id)

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

        Also fills zeros to the charge_model entry until the length resembles grappa.constants.MAX_NUM_CHARGE_MODELS
        """
        if len(self.graphs) == 0:
            return
        
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
            if 'charge_model' in self.graphs[i].nodes['n1'].data.keys() and self.graphs[i].nodes['n1'].data['charge_model'].shape[-1] < constants.MAX_NUM_CHARGE_MODELS:
                self.graphs[i].nodes['n1'].data['charge_model'] = torch.cat(
                    (self.graphs[i].nodes['n1'].data['charge_model'],
                     torch.zeros((self.graphs[i].nodes['n1'].data['charge_model'].shape[0], constants.MAX_NUM_CHARGE_MODELS - self.graphs[i].nodes['n1'].data['charge_model'].shape[-1]))
                     ), dim=-1
                    )
            keep = keep.intersection(set(self.graphs[i].ndata.keys()))

        for graph in self.graphs:
            removed = removed.union(set(graph.ndata.keys()).difference(keep))
            for feature in set(graph.ndata.keys()).difference(keep):
                del graph.ndata[feature]


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
    

    def subsampled(self, factor:float=1., seed:int=0):
        """
        Subsample the dataset by a factor. 1 is the whole dataset, 0 an empty dataset. The subsampling is done in a deterministic way, i.e. the same subsample will be returned for the same seed.
        """
        if factor is None:
            return self
        if factor == 1.:
            return self

        logging.info(f'Sampling {round(factor*100, 2)}% of the original dataset...')
        np.random.seed(seed)
        n = len(self.graphs)
        perm = np.random.permutation(n)
        subsample = perm[:int(n*factor)]
        return self.where(condition=[i in subsample for i in range(n)])


    def create_reference(self, ref_terms:List[str], ff_lookup:Dict[str,str], cleanup:bool=False):
        """
        Stores QM - MM reference energies and gradients as energy_ref and gradient_ref in the graphs.
        Then deletes all other energy and gradient data except for the reference data.
        Args:
            ref_terms (List[str]): list of terms that should be used for the reference data. E.g. ['bond', 'angle', 'dihedral']
            ff_lookup (Dict[str,str]): dictionary mapping dataset names to force field names. If a dataset name is not present in the lookup, the force field is determined automatically by selecting the force field that provides all necessary terms, or, if this is not unique, by selecting the force field 'reference_ff' if present.
            cleanup (bool): if True, all energy and gradient data except for the reference data is deleted from the graphs.
        """
        for dsname, g in zip(self.subdataset, self.graphs):

            # if the dataset name is present in the lookup, use the reference from there
            if dsname in ff_lookup.keys():
                ff = ff_lookup[dsname]
            # otherwise, check the number of force fields for which data exists
            else:
                all_energy_keys = [k for k in g.nodes['g'].data.keys() if 'energy_' in k]
                # assume that the keys have the form energy_ff_contrib
                ffs = set(['_'.join(k.split('_')[1:-1]) for k in all_energy_keys])
                # filter those out for which not all contribs needed are present:
                ffs = [r for r in ffs if all([f'energy_{r}_{t}' in all_energy_keys for t in ref_terms])]
                if len(ffs) == 1:
                    ff = ffs[0]
                elif 'reference_ff' in ffs:
                    ff = 'reference_ff'
                else:
                    raise ValueError(f'Could not uniquely determine force field for dataset {dsname}. Found {ffs}.')
                
            if not all([f'energy_{ff}_{t}' in g.nodes['g'].data.keys() for t in ref_terms]):
                raise ValueError(f'Could not find all reference energies for force field {ff} in dataset {dsname}. and ref terms {ref_terms}. Energies present: {[k for k in g.nodes["g"].data.keys() if "energy_" in k]}')
            
            if not all([f'gradient_{ff}_{t}' in g.nodes['n1'].data.keys() for t in ref_terms]) and 'gradient_qm' in g.nodes['n1'].data.keys():
                raise ValueError(f'Could not find all reference gradients for force field {ff} in dataset {dsname}. and ref terms {ref_terms}. Energies present: {[k for k in g.nodes["n1"].data.keys() if "gradient_" in k]}')
            
            # assign the reference data to the graph
            ref_energy_contribution = torch.sum(torch.stack([g.nodes['g'].data[f'energy_{ff}_{t}' ] for t in ref_terms], dim=0), dim=0)
            g.nodes['g'].data['energy_ref'] = g.nodes['g'].data['energy_qm'] - ref_energy_contribution
            if 'gradient_qm' in g.nodes['n1'].data.keys():
                g.nodes['n1'].data['gradient_ref'] = g.nodes['n1'].data['gradient_qm'] - torch.sum(torch.stack([g.nodes['n1'].data[f'gradient_{ff}_{t}'] for t in ref_terms], dim=0), dim=0)

            if cleanup:
                # now delete all energy and gradient data except for the reference data:
                for k in list(g.nodes['g'].data.keys()):
                    if 'energy_' in k and not k == 'energy_ref':
                        del g.nodes['g'].data[k]
                for k in list(g.nodes['n1'].data.keys()):
                    if 'gradient_' in k and not k == 'gradient_ref':
                        del g.nodes['n1'].data[k]

    
    # since all confs are returned in the test dataloader, batches could become too large for the GPU memory. Therefore, we restrict the number of conformations to val_conf_strategy and split individual molecules into multiple entries if necessary
    def apply_max_confs(self, confs:Union[int, str]):

        self.num_mols = None # keep track of original number of molecules

        if confs in ['max', 'all']:
            return self
        
        if isinstance(confs, int):
            self.num_mols = {}
            new_graphs, new_mol_ids, new_subdatasets = [], [], []
            for (g, dsname, mol_id) in zip(self.graphs, self.subdataset, self.mol_ids):
                self.num_mols[dsname] = self.num_mols.get(dsname, 0) + 1
                if g.nodes['g'].data['energy_qm'].flatten().shape[0] > confs:
                    num_graphs = g.nodes['g'].data['energy_qm'].flatten().shape[0] // confs
                    base_graphs = [copy.deepcopy(g) for _ in range(num_graphs)]
                    # split the tensors with conf dimension:
                    conf_entries = [('n1', 'xyz')]
                    for feat in g.nodes['g'].data.keys():
                        if feat.startswith('energy_'):
                            conf_entries.append(('g', feat))
                    for feat in g.nodes['n1'].data.keys():
                        if feat.startswith('gradient_'):
                            conf_entries.append(('n1', feat))
                    for lvl, feat in conf_entries:
                        base_graphs[0].nodes[lvl].data[feat] = base_graphs[0].nodes[lvl].data[feat][:, :confs]
                        base_graphs[-1].nodes[lvl].data[feat] = base_graphs[-1].nodes[lvl].data[feat][:, -(g.nodes[lvl].data[feat].shape[1] % confs):]
                        for i in range(1, num_graphs-1):
                            base_graphs[i].nodes[lvl].data[feat] = g.nodes[lvl].data[feat][:, i*confs:(i+1)*confs]
                        
                    new_graphs += base_graphs
                    new_mol_ids += [mol_id]*num_graphs
                    new_subdatasets += [dsname]*num_graphs
                else:
                    new_graphs.append(g)
                    new_mol_ids.append(mol_id)
                    new_subdatasets.append(dsname)

        self.graphs = new_graphs
        self.mol_ids = new_mol_ids
        self.subdataset = new_subdatasets
        return self        


def clear_tag(tag:str):
    """
    Deletes the dgl dataset with the given tag such that changes in the moldata files are reflected in the dataset.
    """
    path = data_utils.get_data_path()/'dgl_datasets'/tag
    if path.exists():
        import shutil
        shutil.rmtree(path)
        logging.info(f'Deleted dataset at {path}.')
