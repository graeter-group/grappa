import warnings
from typing import Optional
from collections import defaultdict
import torch
import dgl
from dgl import DGLGraph
from typing import Tuple, List, Dict, Union, Set
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
import logging
from copy import deepcopy
import pandas as pd

from grappa.constants import BONDED_CONTRIBUTIONS
from grappa.data.transforms import one_hot_to_idx
from grappa.utils.tuple_indices import get_neighbor_dict

def get_atomic_numbers(g:dgl.DGLGraph)->torch.Tensor:
    return torch.argwhere(g.nodes['n1'].data['atomic_number'] == 1)[:,1] + 1

def get_parameters(g, suffix="", exclude:Tuple[str,str]=[], terms:List[str]=['n2', 'n3', 'n4', 'n4_improper'])->Dict[str,torch.Tensor]:
    """
    Get the parameters of a graph asuming that they are stored at g.nodes[lvl].data[{k}/{eq}+suffix]
    Returns a dictionary with keys {n2_k, n2_eq, n3_k, n3_eq, n4_k, n4_improper_k}.

    Parameters
    ----------
    g : dgl.DGLGraph
        The graph.
    suffix : str, optional
        Suffix of the parameter name, by default ""

    Returns
    -------
    dict
        Dictionary of parameters.
    """
    params = {}
    for lvl, param_name in BONDED_CONTRIBUTIONS:
        if lvl not in terms:
            continue
        if (lvl, param_name) in exclude:
            continue

        if lvl in g.ntypes:
            if f"{param_name}{suffix}" in g.nodes[lvl].data.keys():
                params[f"{lvl}_{param_name}"] = g.nodes[lvl].data[f"{param_name}{suffix}"]
    return params


def get_energies(g:dgl.DGLGraph, suffix="", center=True)->torch.Tensor:
    """
    Get the energies of a non-batched graph in shape (n_conf) assuming that they are stored at g.nodes['g'].data[energy+suffix] with subtracted mean along conformations for each batch.

    Parameters
    ----------
    g : dgl.DGLGraph
        The graph.
    suffix : str, optional
        Suffix of the energy name, by default ""

    Returns
    -------
    dict
        Dictionary of energies.
    """
    # subtract mean along conformations for each batch and each tuple:
    if not g.nodes['g'].data[f'energy{suffix}'].shape[0] == 1:
        raise RuntimeError(f"Expecting shape (1, n_confs). energies may not be batched! encountered shape {en.shape}")

    en = g.nodes['g'].data[f'energy{suffix}']

    if torch.isnan(en).any():
        raise RuntimeError(f"energies are nan: {en.shape}, {en}")
    
    if center:
        en = en - en.mean(dim=1, keepdim=True)

    return en[0]



def get_gradients(g:dgl.DGLGraph, suffix="")->torch.Tensor:
    """
    Get the gradients of a graph in shape (n_atoms, n_confs, 3) assuming that they are stored at g.nodes[lvl].data[gradient+suffix]. The graph sghould not be batched if it contains nans.

    Parameters
    ----------
    g : dgl.DGLGraph
        The graph.
    suffix : str, optional
        Suffix of the gradient name, by default ""

    Returns
    -------
    tensor of shape (n_atoms, n_confs, 3)
    """
    grads = g.nodes['n1'].data[f'gradient{suffix}']

    if torch.isnan(grads).any():
        raise RuntimeError(f"gradients are nan: {grads.shape}, {grads}")
    return grads


def get_gradient_contributions(g:dgl.DGLGraph, suffix="", contributions:List[str]=['bond', 'angle', 'proper', 'improper', 'nonbonded', 'total'], skip_err=False)->Dict[str,torch.Tensor]:
    """
    Get the gradient contributions from different MM terms stored in a graph. The shape is (n_atoms, n_confs, 3) for each contribution.
    """
    grad_dict = {contrib: torch.empty((g.num_nodes('n1'),0,3), device=g.nodes['n1'].data['gradient_ref'].device) for contrib in contributions}
    for contrib in contributions:
        if f"gradient{suffix}_{contrib}" in g.nodes['n1'].data.keys():
            grad_dict[contrib] = g.nodes['n1'].data[f"gradient{suffix}_{contrib}"]

        # some exceptions:
        elif contrib=="nonbonded" and suffix=="":
            # NOTE: grappa doesnt predict its reference contributions, this only works in the vanilla case in which nonbonded is the only reference
            if all([e in g.nodes['n1'].data.keys() for e in [f"gradient_ref", f"gradient_qm"]]):
                grad_dict[contrib] = (g.nodes['n1'].data[f"gradient_qm"] - g.nodes['n1'].data[f"gradient_ref"])
        elif contrib=="total":
            if f"gradient{suffix}" in g.nodes['n1'].data.keys():
                grad_dict[contrib] = g.nodes['n1'].data[f"gradient{suffix}"]

        elif not skip_err:
            raise RuntimeError(f"Gradient contribution {contrib} not found as gradient{suffix}_{contrib} in graph. Keys are: {g.nodes['n1'].data.keys()}")
    return grad_dict


def get_energy_contributions(g:dgl.DGLGraph, suffix="", contributions:List[str]=['bond', 'angle', 'proper', 'improper', 'nonbonded', 'total'], skip_err=False, center=True)->Dict[str,torch.Tensor]:
    """
    Get the energy contributions from different MM terms stored in a graph. The shape is (n_confs) for each contribution.
    """
    en_dict = {contrib: torch.empty((g.num_nodes('g'), 0), device=g.nodes['g'].data['energy_ref'].device) for contrib in contributions}
    for contrib in contributions:
        if f"energy{suffix}_{contrib}" in g.nodes['g'].data.keys():
            en_dict[contrib] = g.nodes['g'].data[f"energy{suffix}_{contrib}"]

        # some exceptions:
        elif contrib=="nonbonded" and suffix=="":
            # NOTE: grappa doesnt predict its reference contributions, this only works in the vanilla case in which nonbonded is the only reference
            if all([e in g.nodes['g'].data.keys() for e in [f"energy_ref", f"energy_qm"]]):
                en_dict[contrib] = (g.nodes['g'].data[f"energy_qm"] - g.nodes['g'].data[f"energy_ref"])
        elif contrib=="total":
            if f"energy{suffix}" in g.nodes['g'].data.keys():
                en_dict[contrib] = g.nodes['g'].data[f"energy{suffix}"]

        elif not skip_err:
            raise RuntimeError(f"Energy contribution {contrib} not found as energy{suffix}_{contrib} in graph. Keys are: {g.nodes['g'].data.keys()}")
    if center:
        for contrib in contributions:
            try:
                en_dict[contrib] = en_dict[contrib] - en_dict[contrib].mean(dim=1, keepdim=True)
            except:
                print(f"Could not center {contrib} energy contributions: {en_dict[contrib]}")
                raise


    return en_dict


def get_parameter_se(g, suffix1="", suffix2="_ref", l=2):
    """
    Get a tensor of squared errors, summed over the atoms of the subgraphs that were used for batching. The shape thus is (n_batch).

    Parameters
    ----------
    g : dgl.DGLGraph
        The graph.
    suffix1 : str, optional
        Suffix of the gradient name, by default ""
    suffix2 : str, optional
    l : int, optional
        The l-norm to use, by default 2

    Returns
    -------
    se_dict : dict
        Dictionary of squared errors and number of parameters.
        se_dict['parameter_se'] :
            parameter_se : tensor of shape (n_batch)
            num_params : tensor of shape (n_batch)
    """
    se_dict = {}
    for lvl, param_name in [("n2","k"), ("n2","eq"), ("n3","k"), ("n3","eq"), ("n4","k")]:
        param_se, num_params = get_parameter_se_lvl(g, lvl, param_name, suffix1, suffix2, l)
        se_dict[f"{lvl}_{param_name}"] = param_se, num_params
    
    return se_dict


def get_parameter_se_lvl(g, lvl, param_name, suffix1="", suffix2="_ref", l=2):
    """
    Get a tensor of squared errors, summed over the atoms of the subgraphs that were used for batching. The shape thus is (n_batch).

    Parameters
    ----------
    g : dgl.DGLGraph
        The graph.
    lvl : str
        The level of the parameter.
    param_name : str
        The name of the parameter.
    suffix1 : str, optional
        Suffix of the gradient name, by default ""
    suffix2 : str, optional
    l : int, optional
        The l-norm to use, by default 2

    Returns
    -------
    parameter_se : tensor of shape (n_batch)
    num_params : tensor of shape (n_batch)
    """
    # we have to use dgl.readout_nodes to pool the parameters like with the gradients because the subgraphs are batched in one single graph
    params = g.nodes[lvl].data[f"{param_name}{suffix1}"]
    params_ref = g.nodes[lvl].data[f"{param_name}{suffix2}"]
    
    num_tuples = g.batch_num_nodes(ntype=lvl).long() # list of number of tuples in each batch
    
    # multiply all shapes except for the zeroth
    num_params_per_tuple = int(torch.tensor(params.shape[1:]).prod(dim=0).item())

    num_params = num_tuples * num_params_per_tuple

    diffs = params - params_ref

    if l==2:
        diffs = torch.square(diffs)
    else:
        diffs = torch.pow(diffs, l)

    g.nodes[lvl].data[f"{param_name}_se{suffix1}{suffix2}"] = diffs

    # implicitly unbatch the graph, then pool along the lvl dimension
    param_se = dgl.readout_nodes(g, op='sum', ntype=lvl, feat=f'{param_name}_se{suffix1}{suffix2}')

    # now sum over all but the zeroth dimension:
    while len(param_se.shape) > 1:
        param_se = torch.sum(param_se, dim=-1)

    return param_se.double(), num_params.long()


def get_tuplewise_energies(g, suffix="", center=False):
    """
    Get the tuplewise energies of a graph asuming that they are stored at g.nodes[lvl].data[energy+suffix]
    Returns a dictionary with keys {n2, n3, n4, n4_improper} and values of shape (num_tuples, num_confs).

    Parameters
    ----------
    g : dgl.DGLGraph
        The graph.
    suffix : str, optional
        Suffix of the parameter name, by default ""

    Returns
    -------
    dict
        Dictionary of parameters.
    """
    energies = {}
    for lvl in ["n2", "n3", "n4", "n4_improper"]:
        energies[lvl] = g.nodes[lvl].data[f'energy{suffix}']

        if center:
            # subtract mean along conformations for each batch and each tuple:
            energies[lvl] = energies[lvl] - energies[lvl].mean(dim=1, keepdim=True)

    return energies


def get_param_statistics(loader:"GraphDataLoader", suffix="_ref")->Dict[str,Dict[str,torch.Tensor]]:
    '''
    Returns a dictionary with keys {n2_k, n2_eq, n3_k, n3_eq, n4_k, n4_improper_k}. Ignores nan parameters.
    '''
    parameters = None

    with torch.no_grad():
        for g,_ in loader:
            if parameters is None:
                parameters = get_parameters(g,suffix=suffix)
            else:
                these_params = get_parameters(g,suffix=suffix)
                for k, v in these_params.items():
                    if torch.isnan(v).any():
                        continue
                    # remove the suffix from the key in the stat dict:
                    parameters[k.replace(suffix, "")] = torch.cat((parameters[k.replace(suffix, "")], v), dim=0)
        
        param_statistics = {'mean':{}, 'std':{}}

        if parameters is None:
            logging.warning("No MM parameters found in loader. Returning default statistics.")
            return get_default_statistics()

        for k, v in parameters.items():
            param_statistics['mean'][k] = torch.mean(v[torch.where(~torch.isnan(v))[0]], dim=0)
            param_statistics['std'][k] = torch.std(v[torch.where(~torch.isnan(v))[0]], dim=0)

            if len(v[torch.where(~torch.isnan(v))[0]]) > 0:
                assert not torch.isnan(param_statistics['mean'][k]).any(), f"mean of {k} is nan"
                assert not torch.isnan(param_statistics['std'][k]).any(), f"std of {k} is nan"


    # if there are nans, replace them with the default statistics:
    for m in ['mean', 'std']:
        for k, v in param_statistics[m].items():
            if torch.isnan(v).any():
                param_statistics[m][k] = get_default_statistics()[m][k]
                logging.warning(f"Found nan in train MM parameter statistics {m} of {k}. Replacing with default statistics.")

    return param_statistics


def get_default_statistics():
    """
    Just some example statistics obtained at some point in time from a peptide dataset. Better than nothing.
    """
    DEFAULT_STATISTICS = {
    'mean':
        {'n2_k': torch.Tensor([763.2819]), 'n2_eq': torch.Tensor([1.2353]), 'n3_k': torch.Tensor([105.6576]), 'n3_eq': torch.Tensor([1.9750]), 'n4_k': torch.Tensor([ 1.5617e-01, -5.8312e-01,  7.0820e-02, -6.3840e-04,  4.7139e-04, -4.1655e-04]), 'n4_improper_k': torch.Tensor([ 0.0000, -2.3933,  0.0000])},
    'std':
        {'n2_k': torch.Tensor([161.2278]), 'n2_eq': torch.Tensor([0.1953]), 'n3_k': torch.Tensor([26.5965]), 'n3_eq': torch.Tensor([0.0917]), 'n4_k': torch.Tensor([0.4977, 1.2465, 0.1466, 0.0192, 0.0075, 0.0066]), 'n4_improper_k': torch.Tensor([0.0000, 4.0571, 0.0000])}}
    return DEFAULT_STATISTICS



def cannot_be_isomorphic(graph1:nx.Graph, graph2:nx.Graph)->bool:
    """
    Check whether the set of elements in the two graphs is different. If it is, the graphs cannot be isomorphic.
    """
    atomic_numbers1 = graph1.nodes(data='atomic_number')
    atomic_numbers2 = graph2.nodes(data='atomic_number')
    if len(atomic_numbers1) != len(atomic_numbers2):
        return True
    atomic_numbers1 = np.argmax(np.array([node[1].numpy() for node in atomic_numbers1]), axis=-1)
    atomic_numbers2 = np.argmax(np.array([node[1].numpy() for node in atomic_numbers2]), axis=-1)

    if not set(atomic_numbers1) == set(atomic_numbers2):
        return True

def get_isomorphisms(graphs1:List[dgl.DGLGraph], graphs2:List[dgl.DGLGraph]=None, silent:bool=False)->Set[Tuple[int,int]]:
    """
    Returns a set of pairs of indices of isomorphic graphs in the two lists of graphs. If only one list is provided, it will return the isomorphisms within that list.
    This function can be used to validate generated datasets and to assign a consistent mol_id.
    """
    homgraphs1 = [dgl.node_type_subgraph(graph, ['n1']) for graph in tqdm(graphs1, desc="Creating homgraphs",disable=silent)]
    homgraphs2 = [dgl.node_type_subgraph(graph, ['n1']) for graph in tqdm(graphs2, desc="Creating homgraphs",disable=silent)] if graphs2 is not None else homgraphs1

    nx_graphs1 = list([graph.to_networkx(node_attrs=['atomic_number']) for graph in tqdm(homgraphs1, desc="Converting to nx",disable=silent)])
    nx_graphs2 = list([graph.to_networkx(node_attrs=['atomic_number']) for graph in tqdm(homgraphs2, desc="Converting to nx",disable=silent)]) if graphs2 is not None else nx_graphs1

    # for each graph, check how many graphs are isomorphic to it, based on the element stored in graph.ndata['atomic_number']
    if graphs2 is None:
        pairs = [(i, j) for i in tqdm(range(len(nx_graphs1)), desc="Creating pairs",disable=silent) for j in range(i+1, len(nx_graphs1)) if not cannot_be_isomorphic(nx_graphs1[i], nx_graphs1[j])]
    else:
        pairs = [(i, j) for i in tqdm(range(len(nx_graphs1)), desc="Creating pairs",disable=silent) for j in range(len(nx_graphs2)) if not cannot_be_isomorphic(nx_graphs1[i], nx_graphs2[j])]
    # randomly shuffle the pairs to avoid bias in remaining time prediction
    random.shuffle(pairs)
    isomorphic_pairs = []

    def node_match(n1, n2):
        return np.all(n1['atomic_number'].numpy() == n2['atomic_number'].numpy())

    for i, j in tqdm(pairs, desc="Checking isomorphisms",disable=silent):
        if len(nx_graphs1[i].nodes) != len(nx_graphs2[j].nodes):
            continue
        if nx.is_isomorphic(nx_graphs1[i], nx_graphs2[j], node_match=node_match):
            isomorphic_pairs.append((i, j))

    return set(isomorphic_pairs)

def get_isomorphic_permutation(graph1: dgl.DGLGraph, graph2: dgl.DGLGraph) -> List[int]:
    """
    Returns a permutation list to reorder the atoms of `graph2` to match `graph1` 
    based on their isomorphic structure such that e.g.:

    permutation = get_isomorphic_permutation(g1, g2)
    g2.xyz[:,permutation] = g1.xyz

    Parameters:
    - graph1: A DGLGraph object representing the first molecular graph.
    - graph2: A DGLGraph object representing the second molecular graph.

    Returns:
    - List[int]: A list of indices representing the permutation required to align graph1 to graph2.
    """
    homgraph1 = dgl.node_type_subgraph(graph1, ['n1'])
    homgraph2 = dgl.node_type_subgraph(graph2, ['n1'])

    # Convert DGLGraphs to NetworkX graphs for isomorphism checking
    nx_graph1 = homgraph1.to_networkx(node_attrs=['atomic_number'])
    nx_graph2 = homgraph2.to_networkx(node_attrs=['atomic_number'])
    
    def node_match(n1, n2):
        return np.all(n1['atomic_number'].numpy() == n2['atomic_number'].numpy())
    
    # Find the isomorphism mapping between graph1 and graph2 nodes
    gm = nx.isomorphism.GraphMatcher(nx_graph1, nx_graph2, node_match=node_match)
    
    if gm.is_isomorphic():
        # Extract the node correspondence mapping from graph1 to graph2
        mapping = gm.mapping
        # Generate the permutation list based on the mapping
        permutation = [mapping[i] for i in range(len(homgraph1.nodes()))]
        return permutation
    else:
        raise ValueError("Graphs are not isomorphic")


def as_nx(graph:dgl.DGLGraph)->nx.Graph:
    """
    Convert a DGLGraph representing a grappa molecule to a NetworkX graph.
    """
    homgraph = deepcopy(dgl.node_type_subgraph(graph, ['n1']))
    homgraph.ndata['atomic_number'] = torch.argmax(homgraph.ndata['atomic_number'], dim=-1) + 1
    return homgraph.to_networkx(node_attrs=['atomic_number'])

def get_grappa_contributions(tag: str,  min_grappa_atoms: dict = {"n2": 1, "n3": 1, "n4": 1, "n4_improper": 1}) -> dict:
    """
    Compute the contribution of grappa parameters for the interactions in the dataset.

    Args:
        tag (str): The dataset tag.
        min_grappa_atoms (dict): The minimum number of grappa atoms per interaction. Default is {"n2": 1, "n3": 1, "n4": 1, "n4_improper": 1}.
    """
    from grappa.data import Dataset
    
    term_to_interaction = {"n2": "bonds", "n3": "angles", "n4": "propers", "n4_improper": "impropers"}
    
    ds = Dataset.from_tag(tag)

    # Set default grappa contribution to 0
    grappa_contribution = defaultdict(int)

    for graph, _ in ds: 
        try: 
            is_grappa_atom = graph.nodes["n1"].data["grappa_atom"].to(torch.bool)
            grappa_contribution["grappa_atoms"] += is_grappa_atom.sum().item()
            grappa_contribution["trad_atoms"] += (~is_grappa_atom).sum().item()
        except KeyError:
            warnings.warn("No grappa_atom attribute found in the graph. Assuming all atoms are traditional.")
            grappa_contribution["grappa_atoms"] += 0
            grappa_contribution["trad_atoms"] += graph.num_nodes("n1")
        for term in ["n2", "n3", "n4", "n4_improper"]:
            interaction = term_to_interaction[term]
            try: 
                is_grappa_interaction = graph.nodes[term].data["num_grappa_atoms"] >= min_grappa_atoms[term] 
                grappa_contribution[f"grappa_{interaction}"] += is_grappa_interaction.sum().item()
                grappa_contribution[f"trad_{interaction}"] += (~is_grappa_interaction).sum().item()
            except KeyError:
                warnings.warn(f"No num_grappa_atoms attribute found for the {term} nodes in the graph. Assuming all {interaction} are traditional.")
                grappa_contribution[f"grappa_{interaction}"] += 0
                grappa_contribution[f"trad_{interaction}"] += len(graph.nodes[term].data["idxs"])
    for term in ["atoms", "bonds", "angles", "propers", "impropers"]:
        grappa_contribution[f"{term}_total"] = grappa_contribution[f"grappa_{term}"] + grappa_contribution[f"trad_{term}"]
        grappa_contribution[f"{term}_contrib"] = grappa_contribution[f"grappa_{term}"] / grappa_contribution[f"{term}_total"]
        
    return grappa_contribution


def get_grappa_contributions_from_tags(tags: list, min_grappa_atoms: dict = {"n2": 1, "n3": 1, "n4": 1, "n4_improper": 1}) -> pd.DataFrame:
    """
    Compute the contribution of grappa parameters for the interactions in the datasets.

    Args:
        tags (list): List of dataset tags.
        min_grappa_atoms (dict): The minimum number of grappa atoms per interaction. Default is {"n2": 1, "n3": 1, "n4": 1, "n4_improper": 1}.
    """

    grappa_contributions = []
    for t in tags:
        c = get_grappa_contributions(tag=t, min_grappa_atoms=min_grappa_atoms, print_contributions=False)
        grappa_contributions.append(c)
    contribution = pd.DataFrame(grappa_contributions)
    contribution["grappa_contrib"] = (contribution["grappa_bonds"] + contribution["grappa_angles"] + contribution["grappa_propers"] + contribution["grappa_impropers"]) / (contribution["bonds_total"] + contribution["angles_total"] + contribution["propers_total"] + contribution["impropers_total"])
    contribution["grappa_contrib_mean"] = contribution[["bonds_contrib", "angles_contrib", "propers_contrib", "impropers_contrib"]].mean(axis=1)
    contribution["grappa_contrib_std"] = contribution[["bonds_contrib", "angles_contrib", "propers_contrib", "impropers_contrib"]].std(axis=1)
    # contribution["grappa_contrib_diff"] = (contribution["grappa_contrib_mean"] - contribution["grappa_contrib"]).abs()
    return contribution


def get_neighbor_atomic_numbers_dict(graph: DGLGraph, neighbor_dict: dict) -> dict:
    """
    Get the atomic numbers of the neighbors of each atom in the graph.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        neighbor_dict (dict): A dictionary containing the neighbors of each atom.
    """
    atomic_numbers = graph.nodes["n1"].data["atomic_number"]
    neighbor_atomic_numbers_dict = {}
    for i, neighbors in neighbor_dict.items():
        neighbor_atomic_numbers = []
        for neighbor in neighbors:
            atomic_number_one_hot = atomic_numbers[neighbor]
            atomic_number = one_hot_to_idx(atomic_number_one_hot).item() + 1
            neighbor_atomic_numbers.append(atomic_number)
        neighbor_atomic_numbers_dict[i] = neighbor_atomic_numbers
    return neighbor_atomic_numbers_dict


def is_hydrogen_atom(graph: DGLGraph, atom_idx: int) -> bool:
    """
    Check if the atom is a hydrogen atom.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        atom_idx (int): The index of the atom.
    """
    return bool(graph.nodes["n1"].data["atomic_number"][atom_idx, 0].item())


def is_carbon_atom(graph: DGLGraph, atom_idx: int) -> bool:
    """
    Check if the atom is a carbon atom.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        atom_idx (int): The index of the atom.
    """
    return bool(graph.nodes["n1"].data["atomic_number"][atom_idx, 5].item())


def is_nitrogen_atom(graph: DGLGraph, atom_idx: int) -> bool:
    """
    Check if the atom is a nitrogen atom.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        atom_idx (int): The index of the atom.
    """
    return bool(graph.nodes["n1"].data["atomic_number"][atom_idx, 6].item())


def is_oxygen_atom(graph: DGLGraph, atom_idx: int) -> bool:
    """
    Check if the atom is an oxygen atom.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        atom_idx (int): The index of the atom.
    """
    return bool(graph.nodes["n1"].data["atomic_number"][atom_idx, 7].item())


def is_carbonyl_carbon_atom(graph: DGLGraph, atom_idx: int, neighbor_atomic_numbers_dict: dict) -> bool:
    """
    Check if the atom is a carbonyl carbon atom.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        atom_idx (int): The index of the atom.
        neighbor_dict (dict): A dictionary containing the neighbors of each atom.
    """
    neighbor_atomic_numbers = neighbor_atomic_numbers_dict[atom_idx]
    return is_carbon_atom(graph, atom_idx) and len(neighbor_atomic_numbers) == 3 and neighbor_atomic_numbers.count(8) == 1


def is_neighbor_carbonyl_carbon_atom(graph: DGLGraph, atom_idx: int, neighbor_dict: dict, neighbor_atomic_numbers_dict: dict) -> bool:
    """
    Check if the atom is a neighbor of a carbonyl carbon atom.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        atom_idx (int): The index of the atom.
        neighbor_dict (dict): A dictionary containing the neighbors of each atom.
        neighbor_atomic_numbers_dict (dict): A dictionary containing the atomic numbers of the neighbors of each atom.
    """
    return bool(sum([is_carbonyl_carbon_atom(graph, atom, neighbor_atomic_numbers_dict) for atom in neighbor_dict[atom_idx]]))


def are_connected(atoms: list, neighbor_dict: Optional[dict] = None, graph: Optional[DGLGraph]=None) -> bool:
    """
    Check if the given atoms form a connected subgraph.

    Args:
        atoms (List[int]): A list of atom indices.
        neighbor_dict (Optional[dict]): A dictionary mapping each atom to its neighbors. If None, it will be generated.
        graph (Optional[DGLGraph]): The DGLGraph object representing the molecular graph. Required if neighbor_dict is None.
    """
    assert graph is not None or neighbor_dict is not None, "Either graph or neighbor_dict must be provided."

    if neighbor_dict is None:
        neighbor_dict = get_neighbor_dict(graph.nodes["n2"].data["idxs"].tolist())

    atom_set = set(atoms)  # Convert to set for O(1) lookup
    visited = set()

    def dfs(atom):
        """Depth-first search (DFS) to explore the connected component."""
        stack = [atom]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                stack.extend(neighbor for neighbor in neighbor_dict[current] if neighbor in atom_set)

    # Start DFS from the first atom in the list
    dfs(atoms[0])

    return visited == atom_set


def get_connected_atoms(atom_idx: int, neighbor_dict: dict, forbidden: list) -> list:
    """
    Get the indices of all atoms connected to the atom.
    
    Args:
        atom_idx (int): The index of the atom.
        neighbor_dict (dict): A dictionary containing the neighbors of each atom.
        forbidden (list): A list of atom indexes that should not be included in the connected atoms.
    """

    connected_atoms = []
    neighbor = neighbor_dict[atom_idx]

    for atom in neighbor:
        if atom not in forbidden and atom not in connected_atoms:
            connected_atoms.append(atom)
            neighbor.extend(neighbor_dict[atom])
            
    return connected_atoms


def get_methly_carbon_atom(graph: DGLGraph, neighbor_atomic_numbers_dict: dict) -> list:
    """
    Get the indices of the carbon atoms in a methyl group.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        neighbor_atomic_numbers_dict (dict): A dictionary containing the atomic numbers of the neighbors of each atom.
    """
    methyl_carbon_atoms = []
    total_atoms = graph.num_nodes("n1")
    for idx in range(total_atoms):
        if is_carbon_atom(graph, idx):
            neighbor_atomic_numbers = neighbor_atomic_numbers_dict[idx]
            if len(neighbor_atomic_numbers) == 4 and neighbor_atomic_numbers.count(1) == 3:
                methyl_carbon_atoms.append(idx)      
    return methyl_carbon_atoms


def get_ace_carbonly_carbon_atom(graph: DGLGraph, neighbor_dict: dict, neighbor_atomic_numbers_dict: dict) -> int:
    """
    Get the indix of the carbonyl carbon atom in the acetyl cap.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        neighbor_dict (dict): A dictionary containing the neighbors of each atom.
        neighbor_atomic_numbers_dict (dict): A dictionary containing the atomic numbers of the neighbors of each atom.
    """
    ace_carbonyl_carbon_atom = []
    methyl_carbon_atoms = get_methly_carbon_atom(graph, neighbor_atomic_numbers_dict)
    assert methyl_carbon_atoms, "No methyl carbon atom found in the molecule."

    for idx in methyl_carbon_atoms:
        neighbor_atomic_numbers = neighbor_atomic_numbers_dict[idx]
        if neighbor_atomic_numbers.count(6) == 1:
            carbon_atom = neighbor_dict[idx][neighbor_atomic_numbers.index(6)]
            if is_carbonyl_carbon_atom(graph, carbon_atom, neighbor_atomic_numbers_dict):
                ace_carbonyl_carbon_atom.append(carbon_atom)

    assert len(ace_carbonyl_carbon_atom) == 1, f"{len(ace_carbonyl_carbon_atom)} carbonyl carbon atoms found in the molecule."
    return ace_carbonyl_carbon_atom[0]         


def get_nterminal_alpha_carbon_atom(graph: DGLGraph, neighbor_dict: dict, neighbor_atomic_numbers_dict: dict) -> int:
    """
    Get the indix of the N-terminal alpha carbon atom.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        neighbor_dict (dict): A dictionary containing the neighbors of each atom.
        neighbor_atomic_numbers_dict (dict): A dictionary containing the atomic numbers of the neighbors of each atom.
    """

    ace_carbonly_carbon_atom = get_ace_carbonly_carbon_atom(graph, neighbor_dict, neighbor_atomic_numbers_dict)

    neighbor_atomic_numbers = neighbor_atomic_numbers_dict[ace_carbonly_carbon_atom]
    amid_nitrogen_atom = neighbor_dict[ace_carbonly_carbon_atom][neighbor_atomic_numbers.index(7)]
    neighbor_atomic_numbers = neighbor_atomic_numbers_dict[amid_nitrogen_atom]
    neighbor = neighbor_dict[amid_nitrogen_atom]
    nterminal_alpha_carbon_atom = [atom for atom, atomic_number in zip(neighbor, neighbor_atomic_numbers) if atomic_number == 6 and atom != ace_carbonly_carbon_atom and is_neighbor_carbonyl_carbon_atom(graph, atom, neighbor_dict, neighbor_atomic_numbers_dict)]
    assert nterminal_alpha_carbon_atom, "No N-terminal alpha carbon atom found in the molecule."
    assert len(nterminal_alpha_carbon_atom) == 1, f"{len(nterminal_alpha_carbon_atom)} N-terminal alpha carbon atoms found in the molecule."
    return nterminal_alpha_carbon_atom[0]


def get_nterminal_carbonly_carbon_atom(graph: DGLGraph, neighbor_dict: dict, neighbor_atomic_numbers_dict: dict) -> int:
    """
    Get the indix of the carbonyl carbon atom in the backbone of the N-terminal amino acid residue.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        neighbor_dict (dict): A dictionary containing the neighbors of each atom.
        neighbor_atomic_numbers_dict (dict): A dictionary containing the atomic numbers of the neighbors of each atom.
    """
    nterminal_alpha_carbon_atom = get_nterminal_alpha_carbon_atom(graph, neighbor_dict, neighbor_atomic_numbers_dict)

    neighbor = neighbor_dict[nterminal_alpha_carbon_atom]
    nterminal_carbonyl_carbon_atom = [atom for atom in neighbor if is_carbonyl_carbon_atom(graph, atom, neighbor_atomic_numbers_dict)]

    assert len(nterminal_carbonyl_carbon_atom) == 1, f"{len(nterminal_carbonyl_carbon_atom)} N-terminal carbonyl carbon atoms found in the molecule."
    return nterminal_carbonyl_carbon_atom[0]


def get_second_amide_nitrogen_atom(graph: DGLGraph, neighbor_dict: dict, neighbor_atomic_numbers_dict: dict, nterminal_carbonyl_carbon_atom: Optional[int]=None) -> int:
    """
    Get the indix of the nitrogen atom in the amide group of the second amino acid.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        neighbor_dict (dict): A dictionary containing the neighbors of each atom.
        neighbor_atomic_numbers_dict (dict): A dictionary containing the atomic numbers of the neighbors of each atom.
    """
    if nterminal_carbonyl_carbon_atom is None:
        nterminal_carbonyl_carbon_atom = get_nterminal_carbonly_carbon_atom(graph, neighbor_dict, neighbor_atomic_numbers_dict)

    neighbor_atomic_numbers = neighbor_atomic_numbers_dict[nterminal_carbonyl_carbon_atom]
    second_amid_nitrogen_atom = neighbor_dict[nterminal_carbonyl_carbon_atom][neighbor_atomic_numbers.index(7)]

    return second_amid_nitrogen_atom


def get_nterminal_side_chain_atoms(graph: DGLGraph) -> list:
    """
    Get the indices of the atoms in side chain of the N-terminal amino acid.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
    """
    neighbor_dict = get_neighbor_dict(bonds=graph.nodes["n2"].data["idxs"].tolist())
    neighbor_atomic_numbers_dict = get_neighbor_atomic_numbers_dict(graph=graph, neighbor_dict=neighbor_dict)

    nterminal_alpha_carbon_atom = get_nterminal_alpha_carbon_atom(graph, neighbor_dict, neighbor_atomic_numbers_dict)

    neighbor_atomic_numbers = neighbor_atomic_numbers_dict[nterminal_alpha_carbon_atom]
    neighbor = neighbor_dict[nterminal_alpha_carbon_atom]

    # Check if the N-terminal amino acid is a glycine
    if neighbor_atomic_numbers.count(1) == 2:
        return neighbor[neighbor_atomic_numbers.index(1)]
    
    nterminal_beta_carbon_atom = [atom for atom in neighbor if is_carbon_atom(graph, atom) and not is_carbonyl_carbon_atom(graph, atom, neighbor_atomic_numbers_dict)]

    assert nterminal_beta_carbon_atom, "No N-terminal beta carbon atom found in the molecule."
    assert len(nterminal_beta_carbon_atom) == 1, f"{len(nterminal_beta_carbon_atom)} N-terminal beta carbon atoms found in the molecule."
    nterminal_beta_carbon_atom = nterminal_beta_carbon_atom[0]

    amid_nitrogen_atom = neighbor[neighbor_atomic_numbers.index(7)]

    nterminal_side_chain_atoms = get_connected_atoms(nterminal_beta_carbon_atom, neighbor_dict, [amid_nitrogen_atom, nterminal_alpha_carbon_atom])

    return nterminal_side_chain_atoms


def get_nterminal_atoms_from_dipeptide(graph: DGLGraph) -> list:
    """
    Get the indices of the atoms in the N-terminal amino acid and the ACE cap from a dipeptide.
    
    Only works for dipeptides!!!
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
    """
    neighbor_dict = get_neighbor_dict(bonds=graph.nodes["n2"].data["idxs"].tolist())
    neighbor_atomic_numbers_dict = get_neighbor_atomic_numbers_dict(graph=graph, neighbor_dict=neighbor_dict)

    nterminal_carbonly_carbon_atom = get_nterminal_carbonly_carbon_atom(graph, neighbor_dict, neighbor_atomic_numbers_dict)
    second_amide_nitrogen_atom = get_second_amide_nitrogen_atom(graph, neighbor_dict, neighbor_atomic_numbers_dict, nterminal_carbonly_carbon_atom)
    return get_connected_atoms(nterminal_carbonly_carbon_atom, neighbor_dict, forbidden=[second_amide_nitrogen_atom])


def get_cterminal_atoms_from_dipeptide(graph: DGLGraph) -> list:
    """
    Get the indices of the atoms in the C-terminal amino acid and the NME cap from a dipeptide.
    
    Only works for dipeptides!!!

    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
    """
    neighbor_dict = get_neighbor_dict(bonds=graph.nodes["n2"].data["idxs"].tolist())
    neighbor_atomic_numbers_dict = get_neighbor_atomic_numbers_dict(graph=graph, neighbor_dict=neighbor_dict)

    nterminal_carbonly_carbon_atom = get_nterminal_carbonly_carbon_atom(graph, neighbor_dict, neighbor_atomic_numbers_dict)
    second_amide_nitrogen_atom = get_second_amide_nitrogen_atom(graph, neighbor_dict, neighbor_atomic_numbers_dict, nterminal_carbonly_carbon_atom)
    return get_connected_atoms(second_amide_nitrogen_atom, neighbor_dict, forbidden=[nterminal_carbonly_carbon_atom])


def get_percentage_of_atoms(graph: DGLGraph, percentage: float, random_sampling: bool=False) -> list:
    """
    Get the atom indices for a given percentage of all atoms in the graph.
    
    Args:
        graph (DGLGraph): The DGLGraph object representing the molecular graph.
        percentage (float): The percentage of atoms to get.
        random_sampling (bool): If True, the atoms are randomly chosen. If False, atoms are selected based on their index and selected atoms are checked for connectivity. Default is False.
    """
    total_atoms = graph.num_nodes("n1")
    num_atoms = int(total_atoms * percentage + 0.5)
    if random_sampling:
        atoms = list(range(total_atoms))
        return random.sample(atoms, num_atoms)
    atoms = list(range(num_atoms))
    if not are_connected(atoms, graph=graph):
        raise ValueError(f"Selected atoms are not connected: {atoms}")
    return atoms
