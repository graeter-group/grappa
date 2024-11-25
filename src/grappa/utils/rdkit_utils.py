import dgl
import numpy as np

from typing import Tuple, Set, Dict, Union, List
from grappa.utils.graph_utils import as_nx


def get_ring_encoding(mol)->np.ndarray:
    """
    Returns a numpy array of shape (n_atoms, 7) that one-hot encodes wheter a given atom is part of a ring (zeroth entry) or not and how large the ring size is.
    """
    return np.array(
            [
                [
                    atom.IsInRing(),
                    atom.IsInRingSize(3),
                    atom.IsInRingSize(4),
                    atom.IsInRingSize(5),
                    atom.IsInRingSize(6),
                    atom.IsInRingSize(7),
                    atom.IsInRingSize(8),
                ]
                for atom in mol.GetAtoms()
            ]
        ).astype(np.float32)


def rdkit_graph_from_bonds(bonds: List[Tuple[int, int]], atomic_numbers: List[int]=None):
    """
    Returns an rdkit molecule for representing the graph structure of the molecule, without chemical details such as bond order, formal charge and stereochemistry.
    Bond indices should be a list of tuples, where each tuple contains the indices of two bonded atoms.
    """
    from rdkit import Chem
    from rdkit.Chem import rdchem

    all_atoms = np.array(bonds).flatten()
    num_atoms = np.max(all_atoms) + 1

    # initialize the molecule
    mol = Chem.RWMol()

    if atomic_numbers is None:
        atomic_numbers = [0] * num_atoms
    else:
        assert len(atomic_numbers) == num_atoms, f"atomic_numbers must have length {num_atoms}, but has length {len(atomic_numbers)}"
    for i in range(num_atoms):
        chem_atom = rdchem.Atom(int(atomic_numbers[i]))
        mol.AddAtom(chem_atom)


    # bond_order 1 used for all bonds, regardless what type they are
    for a1, a2 in bonds:
        mol.AddBond(a1, a2, rdchem.BondType.SINGLE)

    mol = mol.GetMol()

    return mol


def rdkit_graph_from_dgl(dgl_graph):
    """
    Returns an rdkit molecule for representing the graph structure of the molecule, without chemical details such as bond order, formal charge and stereochemistry.
    dgl_graph must be a dgl graph with atomic_number feature.
    """
    nx_graph = as_nx(dgl_graph)
    return rdkit_graph_from_nx(nx_graph)


def rdkit_graph_from_nx(nx_graph):
    """
    Returns an rdkit molecule for representing the graph structure of the molecule, without chemical details such as bond order, formal charge and stereochemistry.
    nx_graph must be a networkx graph with atomic_number feature
    """
    from rdkit import Chem
    from rdkit.Chem import rdchem

    mol = Chem.RWMol()

    for node, data in nx_graph.nodes(data=True):
        atom = rdchem.Atom(data['atomic_number'])
        mol.AddAtom(atom)

    for edge in nx_graph.edges(data=True):
        a1, a2, data = edge
        mol.AddBond(a1, a2, rdchem.BondType.SINGLE)

    mol = mol.GetMol()

    return mol

def draw_mol(mol, filename:str=None):
    """
    Draw the molecule using rdkit's Draw.MolToImage function.
    """
    from rdkit.Chem import Draw
    from PIL import Image

    img = Draw.MolToImage(mol)
    if filename is not None:
        img.save(filename)
    return img


def get_degree(mol) -> np.ndarray:
    """
    Returns the degree of each atom in the molecule one hot encoded. Can be between 1 and 6, i.e. has shape (n_atoms, 6).
    """
    return np.array(
            [
                [
                    atom.GetDegree() == i
                    for i in range(1, 7)
                ]
                for atom in mol.GetAtoms()
            ]
        ).astype(np.float32)



def construct_bonds(mol) -> Set[Tuple]:
    """
    Construct a set containing the index tuples describing bonds
    """
    bonds = set()

    for bond in mol.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()

        # use symmetry, only store each bond once
        if atom1_idx < atom2_idx:
            bond = (atom1_idx, atom2_idx)
            bonds.add(bond)
    return bonds


# =============================================================================
# inspired by openff, translated to rdkit molecule:

def construct_propers(mol)->Union[Tuple[Set[Tuple], Set[Tuple]], List[Set[Tuple]]]:
    """
    Returns propers
    Construct sets containing the index tuples describing proper torsions
    """


    propers = set()

    for atom1 in mol.GetAtoms():
        atom1_idx = atom1.GetIdx()

        for atom2 in atom1.GetNeighbors():
            atom2_idx = atom2.GetIdx()

            for atom3 in atom2.GetNeighbors():
                atom3_idx = atom3.GetIdx()

                if atom1_idx == atom3_idx:
                    continue

                for atom4 in atom3.GetNeighbors():
                    atom4_idx = atom4.GetIdx()

                    if atom4_idx == atom2_idx:
                        continue
                    # Exclude i-j-k-i
                    if atom1_idx == atom4_idx:
                        continue


                    # choose one of the two equivalent (1,2,3,4) and (4,3,2,1) tuples
                    if atom1_idx < atom4_idx:
                        torsion = (atom1_idx, atom2_idx, atom3_idx, atom4_idx)
                        propers.add(torsion)

    return propers



def construct_angles(mol):
    """
    Get the set of angles.
    Index tuples that can be obtained by invariant permutations are removed, if not, they are included.
    """
    angles = set()

    for atom1 in mol.GetAtoms():
        atom1_idx = atom1.GetIdx()

        for atom2 in atom1.GetNeighbors():
            atom2_idx = atom2.GetIdx()

            for atom3 in atom2.GetNeighbors():
                atom3_idx = atom3.GetIdx()

                # only save on of the identical tuples (e.g. (1,2,3) and (3,2,1) are the same)
                if atom1_idx < atom3_idx:
                    angle = (atom1_idx, atom2_idx, atom3_idx)
                    angles.add(angle)
                
    return angles

# =============================================================================

def is_improper_(rd_mol, idxs:Tuple[int,int,int,int], central_atom_id:int=2)->bool:
    """
    Helper function to check whether the given tuple of indices describes an improper torsion.
    Checks whether the given tuple of indices describes an improper torsion.
    We can assume that the tuples describe either a proper or improper torsion.
    We also assume that the idxs correspond to the indices of the rdkit molecule.
    """
    from rdkit.Chem.rdchem import Mol

    assert isinstance(rd_mol, Mol), f"rd_mol must be an rdkit.Chem.rdchem.Mol, but is {type(rd_mol)}" 
    # check whether the central atom is the connected to all other atoms in the rdkit molecule.

    central_atom = rd_mol.GetAtomWithIdx(idxs[central_atom_id])

    # get the neighbor_dict of the central atom
    neighbor_idxs = set([n.GetIdx() for n in central_atom.GetNeighbors()])

    # for each atom in the torsion, check if it's a neighbor of the central atom
    for i, atom_id in enumerate(idxs):
        if i != central_atom_id:  # skip the central atom itself
            if atom_id not in neighbor_idxs:
                # if one of the atoms is not connected to it, return False
                return False

    # if all atoms are connected to the central atom, this is an improper torsion
    return True
