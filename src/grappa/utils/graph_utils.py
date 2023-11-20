import torch
import dgl
from typing import Tuple, List, Dict, Union

def get_parameters(g, suffix="", exclude:Tuple[str,str]=[]):
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
    for lvl, param_name in [("n2","k"), ("n2","eq"), ("n3","k"), ("n3","eq"), ("n4","k"), ("n4_improper","k")]:
        if (lvl, param_name) in exclude:
            continue

        params[f"{lvl}_{param_name}"] = g.nodes[lvl].data[f"{param_name}{suffix}"]

    return params


def get_energies(g, suffix="", center=True):
    """
    Get the energies of a graph in shape (n_batch, n_conf) assuming that they are stored at g.nodes[lvl].data[energy+suffix] with subtracted mean along conformations for each batch.

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
    en = g.nodes['g'].data[f'energy{suffix}']
    if center:
        en = en - en.mean(dim=1, keepdim=True)
    
    return en



def get_gradients(g, suffix=""):
    """
    Get the gradients of a graph in shape (n_atoms_batch, n_confs, 3) assuming that they are stored at g.nodes[lvl].data[gradient+suffix]

    Parameters
    ----------
    g : dgl.DGLGraph
        The graph.
    suffix : str, optional
        Suffix of the gradient name, by default ""

    Returns
    -------
    tensor of shape (n_atoms_batch, n_confs, 3)
    """

    return g.nodes['n1'].data[f'gradient{suffix}']


def get_gradient_se(g, suffix1="", suffix2="_ref", l=2):
    """
    Get a tensor of errors. For the spatial dimension, the 2-norm is used, for the atom dimension the l-norm. It is summed over the atoms of the subgraphs that were used for batching. The shape thus is (n_batch).

    implements:
    \sum_{i=1}^{n_batch} \sum_{j=1}^{n_atoms_batch_i} | \sum_{d=1}^3 (f_{ijd}^{(1)} - f_{ijd}^{(2)})^2 |^l

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
    force_se : tensor of shape (n_batch)
    num_forces : tensor of shape (n_batch)
    """

    num_confs = int(g.nodes['n1'].data[f'gradient{suffix2}'].shape[1])
    num_atoms = g.batch_num_nodes('n1') # list of number of atoms in each batch
    num_forces = num_atoms * num_confs

    diffs = g.nodes['n1'].data[f'gradient{suffix1}'] - g.nodes['n1'].data[f'gradient{suffix2}']

    # the squared diffs (sum over spatial dimension):
    diffs = torch.sum(torch.square(diffs), dim=-1)

    # take the root (spatial 2-norm) and then ^l before summing over the atoms, thus we have ||f1 - f2||_2^l
    if l != 2:
        diffs = torch.pow(diffs, l/2)
    
    g.nodes['n1'].data[f'gradient_se{suffix1}{suffix2}'] = diffs

    # implicitly unbatch the graph, then pool along the atom dimension
    force_se = dgl.readout_nodes(g, op='sum', ntype='n1', feat=f'gradient_se{suffix1}{suffix2}')

    # now also pool the conformation dimension
    force_se = torch.sum(force_se, dim=-1)

    return force_se.double(), num_forces.long()


def get_energy_se(g, suffix1="", suffix2="_ref", l=2):
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
    energy_se : tensor of shape (n_batch)
    num_confs : tensor of shape (n_batch)
    """

    energies = get_energies(g, suffix=suffix1, center=True)
    energies_ref = get_energies(g, suffix=suffix2, center=True)

    
    # sum over the conf dimension
    if l == 2:
        energy_se = torch.sum(torch.square(energies - energies_ref), dim=-1)
    else:
        energy_se = torch.sum(torch.pow(torch.abs(energies - energies_ref), l), dim=-1)

    # num confs is the same for all batches:
    num_confs = torch.full_like(energy_se, fill_value=energies.shape[-1])
                                
    return energy_se.double(), num_confs.long()


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