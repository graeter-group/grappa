import dgl
from typing import List
from dgl import DGLGraph
import torch
import copy
import numpy as np
import copy

def batch(graphs: List[DGLGraph], deep_copies_of_same_n_atoms:bool=False) -> DGLGraph:
    """
    Returns a batched graph in which the 'idxs' feature is updated to reflect the new node indices of n1 nodes.
    The 'atom_id' is unaffected and thus not unique anymore.
    
    Calls the dgl.batch method but also updates the 'idxs' feature.
    Creates deep copies of the idxs features and of graphs that are present multiple times in the batch.
    """
    # Compute the offsets for the 'n1' node type

    n1_offsets = torch.cumsum(
        torch.tensor([0] + [g.num_nodes('n1') for g in graphs[:-1] if len(graphs) > 1]), dim=0
    )

    num_confs = None
    if 'xyz' in graphs[0].nodes['n1'].data:
        num_confs = graphs[0].nodes['n1'].data['xyz'].shape[1]

    batched_graphs = []


    # make deep copies of the idx features
    # then shift them
    if deep_copies_of_same_n_atoms:
        xyz_shapes_set = set([g.nodes['n1'].data['xyz'].shape for g in graphs])
    for graph, offset in zip(graphs, n1_offsets):
        if deep_copies_of_same_n_atoms:
            # If we have the same graph multiple times in the batch, we need to make a deep copy of the whole graph to avoid autograd errors. Thus, create a deep copy if the shape of xyz did already occur (this is not a sufficient but a necessary condition).
            n_atoms = graph.nodes['n1'].data['xyz'].shape
            if n_atoms in xyz_shapes_set:
                graph = copy.deepcopy(graph)

        if num_confs is not None:
            if num_confs != graph.nodes['n1'].data['xyz'].shape[1]:
                raise ValueError(f'All graphs must have the same number of conformations but found {num_confs} and {graph.nodes["n1"].data["xyz"].shape[1]}')
            
        for ntype in ['n2', 'n3', 'n4', 'n4_improper']:
            graph.nodes[ntype].data['idxs'] = copy.deepcopy(graph.nodes[ntype].data['idxs'])
            graph.nodes[ntype].data['idxs'] = graph.nodes[ntype].data['idxs'] + offset
        
        batched_graphs.append(graph)

    return dgl.batch(batched_graphs)


def unbatch(batched_graph: DGLGraph) -> List[DGLGraph]:
    """
    Splits a batched graph back into a list of individual graphs,
    correcting the 'idxs' feature to reflect the original node indices of the 'n1' type.
    Modifies the idxs feature of the graphs in-place!
    Also deletes all dummy conformations.
    """
    subgraphs = dgl.unbatch(batched_graph)
    n1_offsets = torch.cumsum(
        torch.tensor([0] + [g.num_nodes('n1') for g in subgraphs[:-1]]), dim=0
    )

    for subgraph, offset in zip(subgraphs, n1_offsets):
        for ntype in ['n2', 'n3', 'n4', 'n4_improper']:
            subgraph.nodes[ntype].data['idxs'] = subgraph.nodes[ntype].data['idxs'] - offset

    # delete all dummy-confs:
    for i, subgraph in enumerate(subgraphs):
        subgraphs[i] = delete_dummy_confs(subgraph)

    return subgraphs


def delete_dummy_confs(g:DGLGraph)->DGLGraph:
    """
    Returns a graph with dummy conformations removed. This is done by deleting the conformations from the xyz, gradient, and energy features that have is_dummy==1.
    """
    if not 'is_dummy' in g.nodes['g'].data:
        return g
    
    non_dummy_mask = (g.nodes['g'].data['is_dummy'][0] == 0)

    # remove the dummy confs:
    if not all(non_dummy_mask):

        g.nodes['g'].data['is_dummy'] = g.nodes['g'].data['is_dummy'][:, non_dummy_mask]
        
        g.nodes['n1'].data['xyz'] = g.nodes['n1'].data['xyz'][:, non_dummy_mask, :]
        if torch.isnan(g.nodes['n1'].data['xyz']).any():
            raise RuntimeError(f"Found nan in xyz after unbatching")

        for feat in g.nodes['g'].data.keys():
            if 'energy' in feat:
                assert g.nodes['g'].data[feat].shape[0] == 1, "Internal error while unbatching."
                g.nodes['g'].data[feat] = g.nodes['g'].data[feat][:, non_dummy_mask]

        for feat in g.nodes['n1'].data.keys():
            if 'gradient' in feat:
                grad = g.nodes['n1'].data[feat]
                g.nodes['n1'].data[feat] = grad[:, non_dummy_mask, :]

        # now the tuplewise energies stored at the higher order conjugate graphs:
        for lvl in ["n2", "n3", "n4", "n4_improper"]:
            if lvl in g.ntypes:
                for feat in g.nodes[lvl].data.keys():
                    if 'energy' in feat:
                        g.nodes[lvl].data[feat] = g.nodes[lvl].data[feat][:, non_dummy_mask]
    
    return g


def grad_available():
    """
    Recognize a possible autograd context in which the function is called.
    """
    x = torch.tensor([1.], requires_grad=True)
    y = x * 2
    return y.requires_grad # is false if context is torch.no_grad()


def set_number_confs(g:DGLGraph, num_confs:int, seed:int=None):
    """
    Returns a graph with the number of conformations set to num_confs. This is done by either deleting conformations randomly or by copying input data and creating a new feature graph.nodes['g'].data['is_dummy'] which is 1 for dummy conformations and 0 for real conformations. This is reversed upon calling the grappa.utils.dgl_utils.unbatch() function.
    """
    if 'xyz' not in g.nodes['n1'].data:
        # we can assume that there is no conformational data in the graph
        return g

    confs_present = g.nodes['n1'].data['xyz'].shape[1]
    
    if seed is not None:
        torch.manual_seed(seed)
    

    if confs_present == num_confs:
        g.nodes['g'].data['is_dummy'] = torch.zeros((1, num_confs), dtype=torch.float32)
        return g
    
    elif confs_present > num_confs:
        g.nodes['g'].data['is_dummy'] = torch.zeros((1, num_confs), dtype=torch.float32)
        # choose a subset of conformations:
        conf_idxs = torch.randperm(confs_present)[:num_confs]

    else:
        g.nodes['g'].data['is_dummy'] = torch.cat((torch.zeros((1, confs_present), dtype=torch.float32), torch.ones((1, num_confs-confs_present), dtype=torch.float32)), dim=-1)
        # repeat the last conformation until we have num_confs:
        conf_idxs = torch.cat((torch.arange(confs_present), torch.full((num_confs-confs_present,), confs_present-1, dtype=torch.long)))

    g.nodes['n1'].data['xyz'] = g.nodes['n1'].data['xyz'][:, conf_idxs]
    for feat in g.nodes['g'].data.keys():
        if 'energy' in feat:
            g.nodes['g'].data[feat] = g.nodes['g'].data[feat][:, conf_idxs]
            if torch.isnan(g.nodes['g'].data[feat]).any():
                raise RuntimeError(f"Found nan in {feat} after setting number of conformations to {num_confs}")
    for feat in g.nodes['n1'].data.keys():
        if 'gradient' in feat:
            g.nodes['n1'].data[feat] = g.nodes['n1'].data[feat][:, conf_idxs]
    

    return g
