
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




# inspired from espaloma (outdated):


# MIT License

# Copyright (c) 2020 Yuanqing Wang @ choderalab // MSKCC

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.





def relationship_indices_from_mol(
    mol: Mol
) -> Dict[str, torch.Tensor]:
    """
    Get indices for all relationships from an rdkit mol. These are duplicated, which make the processing less efficient. This resembles espaloma behaviour.
    """

    idxs = tuple_indices.get_indices(mol, reduce_symmetry=False)

    return idxs


def from_homogeneous_and_mol(g, mol:Mol)->dgl.DGLGraph:


    # initialize empty dictionary
    hg = {}

    # get adjacency matrix
    a = g.adjacency_matrix()

    # get indices
    idxs = tuple_indices.get_indices(mol=mol, reduce_symmetry=False)

    assert idxs["n1"].shape[0] == mol.GetNumAtoms()
    assert np.all(idxs["n1"] == np.arange(mol.GetNumAtoms()))

    # =========================
    # neighboring relationships
    # =========================
    # NOTE:
    # here we only define the neighboring relationship
    # on atom level
    hg[("n1", "n1_neighbors_n1", "n1")] = tuple_indices.bond_indices(mol=mol, reduce_symmetry=False).T # both directions, transform from (n_bonds, 2) to (2, n_bonds)


    # ======================================
    # nonbonded terms
    # ======================================
    # NOTE: everything is counted twice here
    # nonbonded is where
    # $A = AA = AAA = AAAA = 0$

    # make dense
    a_ = a.to_dense().detach().numpy()

    idxs["nonbonded"] = np.stack(
        np.where(
            np.equal(a_ + a_ @ a_ + a_ @ a_ @ a_, 0.0)
        ),
        axis=-1,
    )

    # onefour is the two ends of torsion
    # idxs["onefour"] = np.stack(
    #     [
    #         idxs["n4"][:, 0],
    #         idxs["n4"][:, 3],
    #     ],
    #     axis=1,
    # )

    idxs["onefour"] = np.stack(
        np.where(
            np.equal(a_ + a_ @ a_, 0.0) * np.greater(a_ @ a_ @ a_, 0.0),
        ),
        axis=-1,
    )


    # ======================================
    # since we do not need neighborhood for levels other than n1, simply create a graph with only self loops:
    # ======================================
    for term in ["n2", "n3", "n4", "n4_improper", "nonbonded", "onefour"]:
        key = (term, f"{term}_edge", term)
        hg[key] = np.stack(
            [
                np.arange(len(idxs[term])),
                np.arange(len(idxs[term])),
            ], axis=0)


    # transform to tuples of tensors:
    hg = {key: (torch.tensor(value[0]).int(), torch.tensor(value[1]).int())  for key, value in hg.items()}
    hg = dgl.heterograph(hg)

    for feat in g.ndata.keys():
        hg.nodes["n1"].data[feat] = g.ndata[feat]
    # include indices in the nodes themselves
    for term in ["n1", "n2", "n3", "n4", "n4_improper", "onefour", "nonbonded"]:
        hg.nodes[term].data["idxs"] = torch.tensor(idxs[term])

    return hg
