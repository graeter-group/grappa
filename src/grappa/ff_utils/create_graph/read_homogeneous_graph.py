# inspired from espaloma:


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


""" 
Build simple graph from RDKit molecule object, containing only geometric data, no chemical information such as bond order, formal charge, etc.
"""
# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def from_openff_toolkit_mol(mol, use_fp:bool=True, max_element:int=26):
    """
    Creates a homogenous dgl graph containing the one-hot encoded elements and, if use_fp, rdkit features. Stored in the feature type 'h0'. To get the elements only, use the feature type 'h0' and slice the first max_element elements.
    """
    import dgl

    # enter bonds
    bonds = list(mol.bonds)
    bonds_begin_idxs = [bond.atom1_index for bond in bonds]
    bonds_end_idxs = [bond.atom2_index for bond in bonds]

    # initialize graph
    g = dgl.graph((bonds_begin_idxs, bonds_end_idxs))

    g = dgl.add_reverse_edges(g)

    n_atoms = mol.n_atoms
    assert n_atoms == g.num_nodes() , f"error initializing the homogeneous graph: inferred {g.num_nodes()} atoms from the given edges but the molecule has {n_atoms}"

    atomic_numbers = torch.Tensor(
        [atom.atomic_number for atom in mol.atoms]
    )

    atomic_numbers = torch.nn.functional.one_hot(atomic_numbers.long(), num_classes=max_element)

    g.ndata["atomic_number"] = atomic_numbers.float()

    if use_fp:
        rd = mol.to_rdkit()

        g.ndata["mass"] = torch.stack([mass(atom) for atom in rd.GetAtoms()], dim=0)
        g.ndata["formal_charge"] = torch.stack([formal_charge(atom) for atom in rd.GetAtoms()], dim=0)
        g.ndata["in_ring"] = torch.stack([in_ring(atom) for atom in rd.GetAtoms()], dim=0)

        MAX_NUM_BONDS = 6
        degrees = torch.Tensor([min(MAX_NUM_BONDS, degree(atom).item()) for atom in rd.GetAtoms()])
        degrees = torch.nn.functional.one_hot(degrees.long(), num_classes=MAX_NUM_BONDS+1)
        g.ndata["degree"] = degrees.float()


    return g
