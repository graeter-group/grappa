
from ..SysWriter import SysWriter

import openmm
from typing import List, Tuple, Dict, Union, Set, Callable

from openmm.unit import Quantity, radians

import torch
import numpy as np

import dgl



def graph_from_topology(
        topology: openmm.app.Topology, 
        classical_ff: openmm.app.ForceField=openmm.app.ForceField("amber99sbildn.xml"), 
        xyz: np.ndarray=None,
        qm_energies: np.ndarray=None,
        qm_gradients: np.ndarray=None,
        get_charges:Callable=None,
        allow_radicals:bool=True,
        radical_indices: List[int]=None,
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

    writer = SysWriter(top=topology, classical_ff=classical_ff, allow_radicals=allow_radicals, radical_indices=radical_indices)
    writer.get_charges = get_charges
    if xyz is not None:
        writer.init_graph(with_parameters=True)
        writer.write_confs(xyz=xyz, qm_energies=qm_energies, qm_gradients=qm_gradients)
    else:
        writer.init_graph(with_parameters=False)

    return writer.graph
    