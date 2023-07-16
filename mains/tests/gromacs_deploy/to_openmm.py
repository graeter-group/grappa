#%%

import json
with open('GrAPPa_input.json', "r") as f:
    data = json.load(f)

#%%
data["radicals"] = []
data["bonds"] += [[29, 35]]

#%%
from typing import List, Tuple
import dgl
from grappa.constants import MAX_ELEMENT

from openmm.app import Topology, Element

def bonds_to_openmm(atoms:List, bonds:List[Tuple[int]], radicals:List[int]=[], ordered_by_res=True)->dgl.DGLGraph:
    """
    atoms: list of tuples of the form (atom_index, residue, atom_name, (sigma, epsilon), atomic_number)
    bonds: list of tuples of the form (atom_index_1, atom_index_2)
    radicals: list of atom indices that are radicals
    max_element: maximum atomic number for one-hot encoding atom numbers.
    Translates the atom indices to internal indices running from 0 to num_atoms-1. Then stores the external indices in the graph such that they can be recovered later.
    atom_idx = g.nodes["n1"].data["external_idx"][internal_idx] gives the external index of the atom with internal index internal_idx.
    """
    atom_types = [a_entry[1] for a_entry in atoms]
    residues = [a_entry[2] for a_entry in atoms]
    atomic_numbers = [a_entry[5] for a_entry in atoms]
    residue_indices = [a_entry[3] for a_entry in atoms]

    # store these as arrays and write them to the parameter dict later:
    epsilons = [a_entry[4][1] for a_entry in atoms]
    sigmas = [a_entry[4][0] for a_entry in atoms]

    external_idxs = [a_entry[0] for a_entry in atoms] # i-th entry is the index of the i-th atom in the molecule

    external_to_internal_idx = {external_idxs[i]:i for i in range(len(external_idxs))} # i-th entry is the list-position of the atom with index i

    bonds = [(external_to_internal_idx[bond[0]], external_to_internal_idx[bond[1]]) for bond in bonds]

    radical_indices = [external_to_internal_idx[radical] for radical in radicals]



    # create a new, empty topology
    openmm_topology = Topology()

    # create a new chain (assuming all residues are in the same chain)
    chain = openmm_topology.addChain()

    if ordered_by_res:
        # create a new residue every time the residue index changes:
        last_res_index = None
    else:
        # store all residue indices:
        residue_indices_ = []

    for atom_idx, (res_index, res, atom_type, atomic_number) in enumerate(zip(residue_indices, residues, atom_types, atomic_numbers)):

        if ordered_by_res:
            # check if we need to start a new residue
            if res_index != last_res_index:
                residue = openmm_topology.addResidue(res, chain)
                last_res_index = res_index
        else:
            if res_index not in residue_indices_:
                residue = openmm_topology.addResidue(res, chain)
                residue_indices_.append(res_index)

        # determine element based on atom type
        # this is just a basic example; you may need to map atom types to elements differently
        element = Element.getByAtomicNumber(atomic_number)

        # add the atom to the current residue
        openmm_topology.addAtom(name=atom_type, element=element, residue=residue, id=atom_idx)

    # add the bonds to the topology:
    atom_list = list(openmm_topology.atoms())

    # Iterate over bond_list and add each bond to the topology
    for bond in bonds:
        atom1 = atom_list[bond[0]]
        atom2 = atom_list[bond[1]]
        openmm_topology.addBond(atom1, atom2)
        
    return openmm_topology

# %%
top = bonds_to_openmm(data["atoms"], data["bonds"], data["radicals"])
# %%
from grappa.ff_utils.create_graph.graph_init import graph_from_topology
g = graph_from_topology(top)
# %%
g.nodes["n2"].data["idxs"]

import torch
b = g.nodes["n2"].data["idxs"]
bondset = set([tuple(b[i].tolist()) for i in range(len(b))])
inputset = set([tuple(data["bonds"][i]) for i in range(len(data["bonds"]))])
# assert bondset == inputset # No, only if me map back to the other indices!
# %%


from grappa.ff_utils.SysWriter import SysWriter

w = SysWriter.from_dict(data)
# %%
w.init_graph(with_parameters=True)

g = w.graph

g.nodes["n2"].data["idxs"]
# %%
g.nodes["n3"].data["eq_ref"]
# %%
