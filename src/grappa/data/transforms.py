from typing import List, Optional
from dgl.transforms import BaseTransform
import dgl
from dgl import DGLGraph
import torch
from torch import Tensor
from random import randint, random


def get_khop_connected_atoms(graph: DGLGraph, khop: int, start_atom: int) -> Tensor:
    """
    Get a group of connected atoms that are k-hops away from a start atom.

    Args:
        graph (DGLGraph): The DGLGraph object.
        khop (int): The number of hops.
        start_atom (int): The atom index to start from.
    """
    try:
        subgraph, _ = dgl.khop_in_subgraph(graph, nodes={'n1': start_atom}, k=khop)
        return subgraph.nodes["n1"].data["ids"]
    except Exception as e:
        print(f"Error finding k-hop subgraph: {e}")
        return torch.tensor([], dtype=torch.int32)

def one_hot_to_idx(one_hot: Tensor) -> Tensor:
    """
    Convert one-hot encoding to index.
    """
    return torch.nonzero(one_hot, as_tuple=True)[0]

def idx_to_one_hot(idx: Tensor, total: int) -> Tensor:
    """
    Convert index to one-hot encoding.
    """
    one_hot = torch.zeros(total, dtype=torch.float32)
    idx = idx.to(dtype=torch.int64)
    one_hot.scatter_(0, idx, 1.0)
    return one_hot

def annotate_num_grappa_atoms_in_interaction(graph: DGLGraph, terms: List[str] = ['n2', 'n3', 'n4', 'n4_improper'], grappa_atom_ids: Optional[Tensor] = None) -> DGLGraph:
    """
    Annotate the number of grappa atoms for each interaction.

    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        terms (List[str]): List of interaction types to annotate. Default is ['n2', 'n3', 'n4', 'n4_improper'].
        grappa_atom_ids (Optional[Tensor]): Tensor containing the indices of grappa atoms. If None, it will be derived from the graph.

    Returns:
        DGLGraph: The graph with annotated number of grappa atoms for each interaction.
    """
    if grappa_atom_ids is None:
        grappa_atom_ids = one_hot_to_idx(graph.nodes["n1"].data["grappa_atom"])
        
    for term in terms:
        num_grappa_atoms = torch.zeros(len(graph.nodes[term].data["idxs"]), dtype=torch.float32)
        for id, interaction in enumerate(graph.nodes[term].data["idxs"]):
            for atom in interaction:
                if atom in grappa_atom_ids:
                    num_grappa_atoms[id] += 1
        graph.nodes[term].data["num_grappa_atoms"] = num_grappa_atoms
    return graph


class AnnotateGrappaAtomsNInteractions(BaseTransform):
    """
    Annotate grappa atoms and grappa interactions in the graph. 
    
    Grappa atoms are choosen as a group of connected atoms that are khops away from a start atom. In deterministic mode, the start atom is always set to 5 and the number of hops to 3. In random mode, either all atoms are marked as grappa atoms with a chance of 20% or a random atom is choosen as start atom and between 1 and 3 hops are performed. All interactions are annotated with number of grappa atoms involved in the interaction.

    Args:
        deterministic (bool): If True, the grappa atoms are deterministically choosen. If False, a random atoms are selected as grappa atoms. Default is False.
    """

    def __init__(self, deterministic: bool = False):
        self.deterministic = deterministic

    def __call__(self, graph: DGLGraph) -> DGLGraph:
        total_atoms = graph.num_nodes("n1")

        if self.deterministic:
            # Check if grappa atoms are already annotated
            if "grappa_atom" in graph.nodes["n1"].data.keys():
                return graph
            grappa_atoms_ids = get_khop_connected_atoms(graph, khop=3, start_atom=5)
        else:
            rand = random()
            if rand < 0.7:
                grappa_atoms_ids = get_khop_connected_atoms(graph, khop=randint(1,3), start_atom=randint(0, total_atoms - 1)) 
            elif rand < 0.8: # Select only preannotated grappa atoms as grappa atoms
                if "grappa_atom" not in graph.nodes["n1"].data.keys(): 
                    raise RuntimeError("Grappa atoms are not annotated.")
                return graph
            else: # Select all atoms as grappa atoms
                grappa_atoms_ids = torch.arange(0, total_atoms, dtype=torch.int64)

        graph.nodes["n1"].data["grappa_atom"] = idx_to_one_hot(grappa_atoms_ids, total_atoms)
        graph = annotate_num_grappa_atoms_in_interaction(graph, grappa_atom_ids=grappa_atoms_ids)
        return graph
   
