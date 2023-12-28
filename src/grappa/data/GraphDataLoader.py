import dgl
from torch.utils.data import DataLoader
from grappa.data import Dataset
from grappa.utils import dgl_utils
import copy
from typing import Dict, List, Union
import torch
import numpy as np
import json

def get_collate_fn(conf_strategy:Union[str,int]='min', deep_copies_of_same_graphs:bool=False):
    """
    Returns a custom collate function for batching DGL graphs.
    Args:
        conf_strategy: Strategy for batching conformations. If int, then all graphs will have the same number of conformations. If 'min', then the minimum number of conformations will be used.
        deep_copies_of_same_graphs: If True, then deep copies of the same graph will be created if it occurs multiple times in the batch. This is necessary to avoid autograd errors when using a weighted random sampler.
    Returns:
        batched_graph: A single batched DGLGraph.
        subdataset_names: List of subdataset names.
    """

    def collate_fn(batch):
        """
        Custom collate function for batching DGL graphs.
        Args:
            batch: A list of tuples where each tuple is (DGLGraph, subdataset_name).
        Returns:
            batched_graph: A single batched DGLGraph.
            subdataset_names: List of subdataset names.
        """
        assert isinstance(batch, list), f"batch must be a list, but got {type(batch)}"
        assert isinstance(batch[0], tuple), f"batch must be a list of tuples, but got {type(batch[0])}"
        assert isinstance(batch[0][0], dgl.DGLGraph), f"batch must be a list of tuples where the first element is a DGLGraph, but got {type(batch[0][0])}"

        graphs, subdataset_names = zip(*batch)

        graphs = list(graphs)

        # first, if necessary, delete all dummy conformations:
        graphs[:] = [dgl_utils.delete_dummy_confs(g) for g in graphs]

        if deep_copies_of_same_graphs:
            # If we have the same graph multiple times in the batch, we need to make a deep copy of the whole graph to avoid autograd errors. Thus, create a deep copy if the shape of xyz did already occur (this is not a sufficient but a necessary condition).
            xyz_shape_set = set([g.nodes['n1'].data['xyz'].shape for g in graphs])
            for i, graph in enumerate(graphs):
                xyz_shape = graph.nodes['n1'].data['xyz'].shape
                if xyz_shape in xyz_shape_set:
                    graphs[i] = copy.deepcopy(graph)

        # now make the graphs have the same number of conformations.
        # there are different possible strategies for this. the simplest is to just take the minimum number of conformations.
        if isinstance(conf_strategy, int):
            n_confs = min(conf_strategy, max([g.nodes['n1'].data['xyz'].shape[1] for g in graphs]))
        elif conf_strategy == 'min':
            n_confs = min([g.nodes['n1'].data['xyz'].shape[1] for g in graphs])
        elif conf_strategy == 'max' or conf_strategy == 'all':
            n_confs = max([g.nodes['n1'].data['xyz'].shape[1] for g in graphs])
        # elif conf_strategy == 'all':
        #     n_confs = graphs[0].nodes['n1'].data['xyz'].shape[1]
        #     assert all([n_confs == g.nodes['n1'].data['xyz'].shape[1] for g in graphs]), "All graphs must have the same number of conformations if conf_strategy='all'"
        elif conf_strategy == 'mean':
            n_confs = int(np.mean([g.nodes['n1'].data['xyz'].shape[1] for g in graphs]))
        else:
            raise ValueError(f"Unknown conf_strategy: {conf_strategy}")

        # graphs[:] = [dgl_utils.set_number_confs(g, n_confs) for g in graphs]
        graphs = [dgl_utils.set_number_confs(g, n_confs) for g in graphs]

        # create a single batched graph:
        batched_graph = dgl_utils.batch(graphs, deep_copies_of_same_n_atoms=False)

        return batched_graph, subdataset_names
    
    return collate_fn



class GraphDataLoader(DataLoader):
    def __init__(self, dataset:Dataset, *args, shuffle=False, weights:Dict[str,float]={}, conf_strategy:Union[str,int]='mean', balance_factor:float=0., **kwargs):
        """
        Custom DataLoader for handling graph data.
        Args:
            dataset: A Dataset object.
            shuffle: Whether to shuffle the data in each epoch.
            weights: Dictionary mapping subdataset names to weights. If a subdataset name is not in the dictionary, it is assigned a weight of 1.0. If a subdataset has e.g. weight factor 2, it is sampled 2 times more often per epoch than it would be with factor 1. The total number of molecules sampled per epoch is unaffected, i.e. some molecules will not be sampled at all if the weight of other datasets is > 1.
            conf_strategy: Strategy for batching conformations, where conformations can be randomly chosen to be copied or deleted. If int, then all graphs will have at most this number of conformations. Available: 'min', 'max', 'mean', 'all' (which is the same as max)
            balance_factor: Parameter between 0 and 1 that balances sampling of the datasets: 0 means that the molecules are sampled uniformly across all datasets, 1 means that the probabilities are re-weighted such that the sampled number of molecules per epoch is the same for all datasets. The weights assigned in 'weights' are multiplied by the weight factor obtained from balancing.
            
            *args, **kwargs: Arguments passed to the torch DataLoader.

        Example of iterating through the DataLoader:
        for batched_graph, subdataset_names in data_loader.to('cuda'):
            # Now batched_graph is on GPU
        """
        assert isinstance(dataset, Dataset), f"dataset must be a Dataset, but got {type(dataset)}"
        assert isinstance(weights, dict), f"weights must be a dict, but got {type(weights)}"
        assert isinstance(conf_strategy, str) or isinstance(conf_strategy, int), f"conf_strategy must be a str or int, but got {type(conf_strategy)}"
        assert balance_factor >= 0 and balance_factor <= 1, f"balance_factor must be between 0 and 1, but got {balance_factor}"

        if shuffle and (len(weights) or balance_factor>0) > 0:
            if balance_factor > 0:
                all_names = [dsname for _, dsname in dataset]
                occurence_ratios = {name: all_names.count(name)/len(all_names) for name in set(all_names)}


            sample_weights = [1. if subdataset_name not in weights.keys() else weights[subdataset_name] for _, subdataset_name in dataset]


            if balance_factor > 0:
                # reweight the weights such that the total number of molecules sampled per epoch is the same for all datasets if balance_factor=1 and nothing happens if balance_factor=0
                balanced_ratio = 1./float(len(occurence_ratios)) # pretend the dataset is already balanced

                # weighted sum between the balanced ratio and the original ratio such that we have the balanced ratio if balance_factor=0 (because nothing will be done then) and the original ratio if balance_factor=1 (because then we want to assign weights ~1/original_ratio)
                ratio_used = {name: float((1.-balance_factor)*balanced_ratio + balance_factor*occurence_ratios[name]) for name in occurence_ratios.keys()}
                balancing_weights = np.array([1./ratio_used[subdataset_name] for _, subdataset_name in dataset])

                sample_weights = np.array(sample_weights)*balancing_weights

                percentage_dict = {name: occurence_ratios[name]/ratio_used[name] for name in occurence_ratios.keys()}
                normalization = sum(list(percentage_dict.values()))
                percentage_dict = {name: percentage_dict[name]/normalization*100. for name in percentage_dict.keys()}


                print(f"Balancing dataset sampling. Per epoch the loader will sample instances from the datasets with the following percentages: {json.dumps(percentage_dict, indent=4)}")

            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

            return super().__init__(dataset, *args, collate_fn=get_collate_fn(conf_strategy=conf_strategy, deep_copies_of_same_graphs=True), sampler=sampler, **kwargs)
        elif len(weights) > 0:
            raise ValueError("Weights are only supported with shuffle=True") 
        else:
            return super().__init__(dataset, *args, collate_fn=get_collate_fn(conf_strategy=conf_strategy), shuffle=shuffle, **kwargs)


    def to(self, device):
        """
        Custom method to move batched data to the specified device.
        """
        for batch in self:
            batched_graph, subdataset_names = batch
            batched_graph = batched_graph.to(device)
            yield batched_graph, subdataset_names
