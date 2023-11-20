import dgl
from torch.utils.data import DataLoader
from grappa.data import Dataset
from grappa.utils import dgl_utils
import copy
from typing import Dict, List, Union
import torch
import numpy as np

def get_collate_fn(conf_strategy:Union[str,int]='min'):
    """
    Returns a custom collate function for batching DGL graphs.
    Args:
        conf_strategy: Strategy for batching conformations. If int, then all graphs will have the same number of conformations. If 'min', then the minimum number of conformations will be used.
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

        # now make the graphs have the same number of conformations.
        # there are different possible strategies for this. the simplest is to just take the minimum number of conformations.
        if isinstance(conf_strategy, int):
            n_confs = conf_strategy
        elif conf_strategy == 'min':
            n_confs = min([g.nodes['n1'].data['xyz'].shape[1] for g in graphs])
        elif conf_strategy == 'max':
            n_confs = max([g.nodes['n1'].data['xyz'].shape[1] for g in graphs])
        elif conf_strategy == 'all':
            n_confs = graphs[0].nodes['n1'].data['xyz'].shape[1]
            assert all([n_confs == graphs[0].nodes['n1'].data['xyz'].shape[1] for g in graphs]), "All graphs must have the same number of conformations if conf_strategy='all'"
        elif conf_strategy == 'mean':
            n_confs = int(np.mean([g.nodes['n1'].data['xyz'].shape[1] for g in graphs]))
        else:
            raise ValueError(f"Unknown conf_strategy: {conf_strategy}")

        # graphs[:] = [dgl_utils.set_number_confs(g, n_confs) for g in graphs]
        graphs = [dgl_utils.set_number_confs(g, n_confs) for g in graphs]

        batched_graph = dgl_utils.batch(graphs) # creates a single batched graph.

        return batched_graph, subdataset_names
    
    return collate_fn



class GraphDataLoader(DataLoader):
    def __init__(self, dataset:Dataset, *args, shuffle=False, weights:Dict[str,float]={}, conf_strategy:Union[str,int]='min', **kwargs):
        """
        Custom DataLoader for handling graph data.
        Args:
            *args, **kwargs: Arguments passed to the torch DataLoader.

        Example of iterating through the DataLoader:
        for batched_graph, subdataset_names in data_loader.to('cuda'):
            # Now batched_graph is on GPU
        """
        if shuffle and len(weights) > 0:
            # raise NotImplementedError("Weighting is not supported yet")
            sample_weights = []
            for g, subdataset_name in dataset:
                if subdataset_name not in weights.keys():
                    sample_weights.append(1.0)
                else:
                    sample_weights.append(weights[subdataset_name])

            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

            return super().__init__(dataset, *args, collate_fn=get_collate_fn(conf_strategy=conf_strategy), sampler=sampler, **kwargs)
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
