import torch
import dgl

def get_parameters(g, suffix=""):
    """
    Get the parameters of a graph asuming that they are stored at g.nodes[lvl].data[{k}/{eq}+suffix]

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
    for node in g.nodes():
        for key in g.nodes[node].data.keys():
            if key.endswith(suffix):
                params[key] = g.nodes[node].data[key]
    return params


def get_energies(g, suffix=""):
    """
    Get the energies of a graph asuming that they are stored at g.nodes[lvl].data[energy+suffix]

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

    return g.nodes['g'].data[f'energy{suffix}']



def get_gradients(g, suffix=""):
    """
    Get the gradients of a graph asuming that they are stored at g.nodes[lvl].data[gradient+suffix]

    Parameters
    ----------
    g : dgl.DGLGraph
        The graph.
    suffix : str, optional
        Suffix of the gradient name, by default ""

    Returns
    -------
    dict
        Dictionary of gradients.
    """

    return g.nodes['n1'].data[f'gradient{suffix}']