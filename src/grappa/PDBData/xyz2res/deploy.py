#%%

from .get_residues import write_residues
import numpy as np

import torch
import dgl
from pathlib import Path
from .model_ import Representation

from .constants import MAX_ELEMENT
from grappa.units import RESIDUES

#%%
"""
Function returning a list of residues and a list of residue numbers, starting from the side of the ACE cap (if present).
the latter can be used ot sort the elements by residue.
xyz must be numpy array of one configuration, i.e. of shape n_atoms x 3.
unit should be angstrom, preferably, might also work for other units.
This is done using a graph neural network to classify c alpha atoms and uses this for algorithmic identification of residues.
"""
def xyz2res(xyz:np.ndarray, atom_numbers:np.ndarray, debug:bool=False):
    import ase
    from ase.geometry.analysis import Analysis
    # make ase graph:
    n = xyz.shape[0]
    pos = xyz
    ase_mol = ase.Atoms(f"N{n}")
    ase_mol.set_positions(pos)
    ase_mol.set_atomic_numbers(atom_numbers)

    if debug:
        print("ase molecule:", ase_mol)

    ana = Analysis(ase_mol)
    connectivity = ana.nl[0].get_connectivity_matrix()

    node_pairs = [(n1,n2) for (n1,n2) in connectivity.keys() if n1!=n2]

    # from this, generate dgl graph:
    node_tensors = [ torch.tensor([node_pair[i] for node_pair in node_pairs], dtype=torch.int32) for i in [0,1] ]

    g = dgl.graph(tuple(node_tensors))
    g = dgl.add_reverse_edges(g)

    assert g.num_nodes() == len(atom_numbers) , f"Number of nodes of the molecular graph ({g.num_nodes()}) must match the length of the atom_numbers array ({len(atom_numbers)}). This might be due to a wrong distance unit in the input. distances must be expressed in angstrom."
    
    g.ndata["atomic_number"] = torch.nn.functional.one_hot(torch.tensor(atom_numbers).long(), num_classes=MAX_ELEMENT)*1.

    # now we have the graph, next, load the model and apply it to the graph
    model = Representation(256,out_feats=1,n_residuals=2,n_conv=1)

    model.load_state_dict(torch.load(Path(__file__).parent / Path("match_model.pt")))

    g = model(g)

    g.ndata["c_alpha"] = torch.where(g.ndata["h"] >= 0, 1, 0)



    g = write_residues(g)

    residue_numbers = g.ndata["res_number"].int().tolist()
    residues = [RESIDUES[res_idx.item()] for res_idx in g.ndata["pred_residue"].int()]

    return residues, residue_numbers