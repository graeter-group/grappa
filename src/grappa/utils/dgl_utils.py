import dgl
from typing import List
from dgl import DGLGraph
import torch


def batch(graphs: List[DGLGraph]) -> DGLGraph:
    """
    Returns a batched graph in which the 'idxs' feature is updated to reflect the new node indices of n1 nodes.
    The 'atom_id' is unaffected and thus not unique anymore.
    
    Calls the dgl.batch method but also updates the 'idxs' feature.
    Modifies the idxs feature of the graphs in-place!
    """
    # Compute the offsets for the 'n1' node type
    n1_offsets = torch.cumsum(
        torch.tensor([0] + [g.num_nodes('n1') for g in graphs[:-1]]), dim=0
    )

    for graph, offset in zip(graphs, n1_offsets):
        for ntype in ['n2', 'n3', 'n4', 'n4_improper']:
            if ntype in graph.ntypes and 'idxs' in graph.nodes[ntype].data:
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
            if ntype in subgraph.ntypes and 'idxs' in subgraph.nodes[ntype].data:
                subgraph.nodes[ntype].data['idxs'] -= offset

    return subgraphs


def grad_available():
    """
    Recognize a possible autograd context in which the function is called.
    """
    x = torch.tensor([1.], requires_grad=True)
    y = x * 2
    return y.requires_grad # is false if context is torch.no_grad()
