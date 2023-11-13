import dgl
from torch.utils.data import DataLoader
from grappa.data import Dataset
from grappa.utils import dgl_utils
import copy


def collate_fn(batch):
    """
    Custom collate function for batching DGL graphs.
    Args:
        batch: A list of tuples where each tuple is (DGLGraph, subdataset_name).
    Returns:
        batched_graph: A single batched DGLGraph.
        subdataset_names: List of subdataset names.
    """
    graphs, subdataset_names = zip(*batch)

    # now make the graphs have the same number of conformations.
    # there are different possible strategies for this. the simplest is to just take the minimum number of conformations.
    # NOTE make the strategy a keyword of the dataloader.
    min_confs = min([g.nodes['n1'].data['xyz'].shape[1] for g in graphs])

    graphs = [dgl_utils.set_number_confs(g, min_confs) for g in graphs]

    batched_graph = dgl_utils.batch(graphs) # creates a single batched graph.
    return batched_graph, subdataset_names



class GraphDataLoader(DataLoader):
    def __init__(self, dataset:Dataset, *args, **kwargs):
        """
        Custom DataLoader for handling graph data.
        Args:
            *args, **kwargs: Arguments passed to the torch DataLoader.

        Example of iterating through the DataLoader:
        for batched_graph, subdataset_names in data_loader.to('cuda'):
            # Now batched_graph is on GPU
        """
        super().__init__(dataset, *args, collate_fn=collate_fn, **kwargs)

    def to(self, device):
        """
        Custom method to move batched data to the specified device.
        """
        for batch in self:
            batched_graph, subdataset_names = batch
            batched_graph = batched_graph.to(device)
            yield batched_graph, subdataset_names
