import dgl
from typing import List
from dgl import DGLGraph
import torch
import copy
import numpy as np

def batch(graphs: List[DGLGraph]) -> DGLGraph:
    """
    Returns a batched graph in which the 'idxs' feature is updated to reflect the new node indices of n1 nodes.
    The 'atom_id' is unaffected and thus not unique anymore.
    
    Calls the dgl.batch method but also updates the 'idxs' feature.
    Modifies the idxs feature of the graphs in-place!
    """
    # Compute the offsets for the 'n1' node type

    n1_offsets = torch.cumsum(
        torch.tensor([0] + [g.num_nodes('n1') for g in graphs[:-1] if len(graphs) > 1]), dim=0
    )

    num_confs = None
    if 'xyz' in graphs[0].nodes['n1'].data:
        num_confs = graphs[0].nodes['n1'].data['xyz'].shape[1]

    batched_graph = graphs

    # make deep copies of the idx features
    # then shift them
    for graph, offset in zip(batched_graph, n1_offsets):
        if num_confs is not None:
            if num_confs != graph.nodes['n1'].data['xyz'].shape[1]:
                raise ValueError(f'All graphs must have the same number of conformations but found {num_confs} and {graph.nodes["n1"].data["xyz"].shape[1]}')
            
        for ntype in ['n2', 'n3', 'n4', 'n4_improper']:
            graph.nodes[ntype].data['idxs'] = copy.deepcopy(graph.nodes[ntype].data['idxs'])
            graph.nodes[ntype].data['idxs'] += offset

    return dgl.batch(graphs)


def unbatch(batched_graph: DGLGraph) -> List[DGLGraph]:
    """
    Splits a batched graph back into a list of individual graphs,
    correcting the 'idxs' feature to reflect the original node indices of the 'n1' type.
    Modifies the idxs feature of the graphs in-place!
    """
    subgraphs = dgl.unbatch(batched_graph)
    n1_offsets = torch.cumsum(
        torch.tensor([0] + [g.num_nodes('n1') for g in subgraphs[:-1]]), dim=0
    )

    for subgraph, offset in zip(subgraphs, n1_offsets):
        for ntype in ['n2', 'n3', 'n4', 'n4_improper']:
            subgraph.nodes[ntype].data['idxs'] -= offset

    return subgraphs


def grad_available():
    """
    Recognize a possible autograd context in which the function is called.
    """
    x = torch.tensor([1.], requires_grad=True)
    y = x * 2
    return y.requires_grad # is false if context is torch.no_grad()


def set_number_confs(g:DGLGraph, num_confs:int, seed:int=None):
    """
    Returns a graph with the number of conformations set to num_confs. This is done by either deleting conformations randomly or by duplicating conformations (sampled with replacement).
    """
    if 'xyz' not in g.nodes['n1'].data:
        # we can assume that there is no conformational data in the graph
        return g

    confs_present = g.nodes['n1'].data['xyz'].shape[1]
    
    if confs_present == num_confs:
        return g
    
    if seed is not None:
        torch.manual_seed(seed)

    if confs_present < num_confs:
        conf_idxs = torch.randint(confs_present, size=(num_confs,), dtype=torch.long)
    else:
        conf_idxs = np.random.choice(np.arange(confs_present), size=(num_confs,), replace=False)
        conf_idxs = torch.tensor(conf_idxs, dtype=torch.long)

    g.nodes['n1'].data['xyz'] = g.nodes['n1'].data['xyz'][:,conf_idxs]
    for feat in g.nodes['g'].data.keys():
        if 'energy' in feat:
            g.nodes['g'].data[feat] = g.nodes['g'].data[feat][:,conf_idxs]
    for feat in g.nodes['n1'].data.keys():
        if 'gradient' in feat:
            g.nodes['n1'].data[feat] = g.nodes['n1'].data[feat][:,conf_idxs]

    return g
