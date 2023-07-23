# rudimentarily inspired from espaloma



""" 
Build simple graph from RDKit molecule object, containing only geometric data, no chemical information such as bond order, formal charge, etc.
"""


from rdkit.Chem.rdchem import Mol
import torch

from . import tuple_indices


def in_ring(atom):
    """
    Returns a feature vector for a given atom.
    """
    return torch.tensor(
                [   
                    atom.IsInRing() * 1.0,
                    atom.IsInRingSize(3) * 1.0,
                    atom.IsInRingSize(4) * 1.0,
                    atom.IsInRingSize(5) * 1.0,
                    atom.IsInRingSize(6) * 1.0,
                    atom.IsInRingSize(7) * 1.0,
                    atom.IsInRingSize(8) * 1.0,
                ],
                dtype=torch.float32,
            )

def mass(atom):
    """
    Returns a feature vector for a given atom.
    """
    return torch.tensor(
                [   
                    atom.GetMass(),
                ],
                dtype=torch.float32,
            )

def formal_charge(atom):
    """
    Returns a feature vector for a given atom.
    """
    return torch.tensor(
                [   
                    atom.GetFormalCharge(),
                ],
                dtype=torch.float32,
            )

def degree(atom):
    """
    Returns a feature vector for a given atom.
    """
    return torch.tensor(
                [   
                    atom.GetTotalDegree(),
                ],
                dtype=torch.float32,
            )

def from_rdkit_mol(mol:Mol, max_element:int=53):
    """
    Creates a homogenous dgl graph containing the one-hot encoded elements and rdkit features. Stored in the feature types "atomic_number", "mass", "formal_charge", "degree", "in_ring" at the 'n1' level of the graph.
    """
    import dgl

    # enter bonds
    bonds = tuple_indices.bond_indices(mol, reduce_symmetry=True)

    # initialize directed graph
    g = dgl.graph((bonds[:,0].tolist(), bonds[:,1].tolist()))

    # make it undirected
    g = dgl.add_reverse_edges(g)

    n_atoms = mol.GetNumAtoms()
    assert n_atoms == g.num_nodes() , f"error initializing the homogeneous graph: inferred {g.num_nodes()} atoms from the given edges but the molecule has {n_atoms}"

    atomic_numbers = torch.Tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    )

    atomic_numbers = torch.nn.functional.one_hot(atomic_numbers.long(), num_classes=max_element)

    g.ndata["atomic_number"] = atomic_numbers.float()

    g.ndata["mass"] = torch.stack([mass(atom) for atom in mol.GetAtoms()], dim=0)
    g.ndata["formal_charge"] = torch.stack([formal_charge(atom) for atom in mol.GetAtoms()], dim=0)
    g.ndata["in_ring"] = torch.stack([in_ring(atom) for atom in mol.GetAtoms()], dim=0)

    # MAX_NUM_BONDS = 6
    # degrees = torch.Tensor([min(MAX_NUM_BONDS, degree(atom).item()) for atom in mol.GetAtoms()])
    # degrees = torch.nn.functional.one_hot(degrees.long(), num_classes=MAX_NUM_BONDS+1)
    # g.ndata["degree"] = degrees.float()


    return g

def from_bonds(bond_idxs:torch.Tensor, max_element:int=53):
    """
    Creates an empty homogenous dgl graph describing only connectivity.
    """
    import dgl

    # initialize directed graph
    g = dgl.graph((bond_idxs[:,0].tolist(), bond_idxs[:,1].tolist()))

    # make it undirected
    g = dgl.add_reverse_edges(g)

    return g
