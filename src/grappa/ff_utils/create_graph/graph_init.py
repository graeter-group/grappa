
from ..SysWriter import SysWriter

import openmm
from typing import List, Tuple, Dict, Union, Set, Callable
from .read_homogeneous_graph import from_bonds
from .read_heterogeneous_graph import from_homogeneous_and_idxs

from grappa.ff_utils.classical_ff.collagen_utility import get_mod_amber99sbildn

from openmm.unit import Quantity, radians

import torch
import numpy as np

import dgl

def graph_from_topology(
        topology: openmm.app.Topology,
        classical_ff: Union[openmm.app.ForceField, str]=get_mod_amber99sbildn(), 
        xyz: np.ndarray=None,
        qm_energies: np.ndarray=None,
        qm_gradients: np.ndarray=None,
        get_charges:Callable=None,
        allow_radicals:bool=True,
        radical_indices: List[int]=None,
        smiles:str=None,
    ) -> dgl.DGLGraph:
    """
    Create a DGLGraph from a topology and a forcefield.
    The forcefield must be able to parametrize the topology.
    The forcefield is used to determine the index tuples for bonds, angles, propers and most importantly (because not unique) impropers.
    If xyz is given, the classical force field parameters are written with suffix "_ref" in the graph.
    If qm_energies or qm_gradients are given, they are written as "u_qm" and "grad_qm" in the graph. Also, reference energies are calculated for training, that is the qm values minus the nonbonded contribution of the classical force field. These are saved as "u_ref" and "grad_ref" in the graph.

    xyz: Array of shape (N_conf x N_atoms x 3) containing the atom positions in grappa.units.
    qm_energies: Array of shape (N_conf) containing the energies in grappa.units.
    qm_gradients: Array of shape (N_conf x N_atoms x 3) containing the gradients in grappa.units.
    """
    if not smiles is None:
        assert isinstance(classical_ff, str), "If smiles is given, classical_ff must be a string."
        assert topology is None, "If smiles is given, topology must be set to None."
        writer = SysWriter.from_smiles(smiles=smiles, ff=classical_ff)
    else:
        writer = SysWriter(top=topology, classical_ff=classical_ff, allow_radicals=allow_radicals, radical_indices=radical_indices)

    writer.get_charges = get_charges
    if xyz is not None:
        writer.init_graph(with_parameters=True)
        writer.write_confs(xyz=xyz, qm_energies=qm_energies, qm_gradients=qm_gradients)
    else:
        writer.init_graph(with_parameters=False)

    return writer.graph
    


def get_empty_graph(bond_idxs:torch.Tensor, angle_idxs:torch.Tensor, proper_idxs:torch.Tensor, improper_idxs:torch.Tensor, use_impropers:bool=True)->dgl.DGLGraph:
    """
    Returns an empty heterograph containing only connectivity induced by bon_indices and noe types g, n1, n2, n3, n4, n4_improper for storing corresponding features later on.
    """
    g = from_bonds(bond_idxs=bond_idxs)
    g = from_homogeneous_and_idxs(g=g, bond_idxs=bond_idxs, angle_idxs=angle_idxs, proper_idxs=proper_idxs, improper_idxs=improper_idxs, use_impropers=use_impropers)

    return g