
import numpy as np
import torch
from . import tuple_indices
import dgl

from rdkit.Chem.rdchem import Mol

from typing import Dict




def from_homogeneous_and_idxs(g:dgl.DGLGraph, bond_idxs:torch.Tensor, angle_idxs:torch.Tensor, proper_idxs:torch.Tensor, improper_idxs:torch.Tensor, use_impropers:bool=True) -> dgl.DGLGraph:

    # initialize empty dictionary
    hg = {}


    idxs = {"n2": bond_idxs, "n3": angle_idxs, "n4": proper_idxs, "n4_improper": improper_idxs}

    for k, v in idxs.items():
        if not v is None:
            idxs[k] = torch.tensor(v)

    assert g.num_edges() == len(bond_idxs)*2, f"number of edges in graph ({g.num_edges()}) does not match 2*number of bonds ({2*len(bond_idxs)})"

    # define the heterograph structure:

    b = idxs["n2"].transpose(0,1) # transform from (n_bonds, 2) to (2, n_bonds)

    first_idxs = torch.cat((b[0], b[1]), dim=0)
    second_idxs = torch.cat((b[1], b[0]), dim=0) # shape (2*n_bonds,)

    hg[("n1", "n1_edge", "n1")] = torch.stack((first_idxs, second_idxs), dim=0).int() # shape (2, 2*n_bonds)
    
    # ======================================
    # since we do not need neighborhood for levels other than n1, simply create a graph with only self loops:
    # ======================================
    
    terms_ = ["n2", "n3", "n4", "n4_improper"] if use_impropers else ["n2", "n3", "n4"]

    TERMS = []
    for t in terms_:
        if not idxs[t] is None:
            if len(idxs[t]) > 0:
                TERMS.append(t)

    for term in TERMS+["g"]:
        key = (term, f"{term}_edge", term)
        n_nodes = len(idxs[term]) if term != "g" else 1
        hg[key] = torch.stack(
            [
                torch.arange(n_nodes),
                torch.arange(n_nodes),
            ], dim=0).int()


    # transform to tuples of tensors:
    hg = {key: (value[0], value[1]) for key, value in hg.items()}

    for k, (vsrc,vdest) in hg.items():
        # make sure that the tensors have the correct shape:
        assert vsrc.shape == vdest.shape, f"shape of {k} is {vsrc.shape} and {vdest.shape}"
        assert len(vdest.shape) > 0, f"shape of {k} is {vdest.shape} and {vdest.shape}"
        assert vsrc.shape[0] > 0, f"shape of {k} is {vsrc.shape} and {vdest.shape}"

    # init graph
    hg = dgl.heterograph(hg)

    for feat in g.ndata.keys():
        hg.nodes["n1"].data[feat] = g.ndata[feat]

    # write indices in the nodes
    for term in TERMS:
        hg.nodes[term].data["idxs"] = idxs[term].int()

    return hg

