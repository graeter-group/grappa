import torch
import dgl
from typing import Tuple, List, Dict, Union, Set
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
import logging

from grappa.constants import BONDED_CONTRIBUTIONS

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
    atomic_numbers1 = np.argmax(np.array([node[1] for node in atomic_numbers1]), axis=-1)
    atomic_numbers2 = np.argmax(np.array([node[1] for node in atomic_numbers2]), axis=-1)

    if not set(atomic_numbers1) == set(atomic_numbers2):
        return True

def get_isomorphisms(graphs1:List[dgl.DGLGraph], graphs2:List[dgl.DGLGraph]=None)->Set[Tuple[int,int]]:
    """
    Returns a set of pairs of indices of isomorphic graphs in the two lists of graphs. If only one list is provided, it will return the isomorphisms within that list.
    This function can be used to validate generated datasets and to assign a consistent mol_id.
    """
    homgraphs1 = [dgl.node_type_subgraph(graph, ['n1']) for graph in tqdm(graphs1, desc="Creating homgraphs")]
    homgraphs2 = [dgl.node_type_subgraph(graph, ['n1']) for graph in tqdm(graphs2, desc="Creating homgraphs")] if graphs2 is not None else homgraphs1

    nx_graphs1 = list([graph.to_networkx(node_attrs=['atomic_number']) for graph in tqdm(homgraphs1, desc="Converting to nx")])
    nx_graphs2 = list([graph.to_networkx(node_attrs=['atomic_number']) for graph in tqdm(homgraphs2, desc="Converting to nx")]) if graphs2 is not None else nx_graphs1

    # for each graph, check how many graphs are isomorphic to it, based on the element stored in graph.ndata['atomic_number']
    if graphs2 is None:
        pairs = [(i, j) for i in tqdm(range(len(nx_graphs1)), desc="Creating pairs") for j in range(i+1, len(nx_graphs1)) if not cannot_be_isomorphic(nx_graphs1[i], nx_graphs1[j])]
    else:
        pairs = [(i, j) for i in tqdm(range(len(nx_graphs1)), desc="Creating pairs") for j in range(len(nx_graphs2)) if not cannot_be_isomorphic(nx_graphs1[i], nx_graphs2[j])]
    # randomly shuffle the pairs to avoid bias in remaining time prediction
    random.shuffle(pairs)
    isomorphic_pairs = []

    def node_match(n1, n2):
        return np.all(n1['atomic_number'].numpy() == n2['atomic_number'].numpy())

    for i, j in tqdm(pairs, desc="Checking isomorphisms"):
        if len(nx_graphs1[i].nodes) != len(nx_graphs2[j].nodes):
            continue
        if nx.is_isomorphic(nx_graphs1[i], nx_graphs2[j], node_match=node_match):
            isomorphic_pairs.append((i, j))

    return set(isomorphic_pairs)
