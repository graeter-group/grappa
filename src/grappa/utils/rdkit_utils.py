
import numpy as np

from typing import Tuple, Set, Dict, Union, List


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
        )


def rdkit_graph_from_bonds(bonds: List[Tuple[int, int]]):
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

    for _ in range(num_atoms):
        chem_atom = rdchem.Atom(0)
        mol.AddAtom(chem_atom)


    # bond_order 1 used for all bonds, regardless what type they are
    for a1, a2 in bonds:
        mol.AddBond(a1, a2, rdchem.BondType.SINGLE)

    mol = mol.GetMol()

    return mol


############################################################################################################
# LEGACY RDKIT METHODS FOR TUPLES


def atom_indices(mol) -> np.ndarray:
    """
    Obtain the indices of atoms in the given molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to get atom indices from.

    Returns:
    np.ndarray: A numpy array containing the indices of atoms in the molecule.
    """
    return np.array([a.GetIdx() for a in mol.GetAtoms()]).astype(np.int64)


def bond_indices(mol) -> np.ndarray:
    """
    Obtain the indices of bonds in the given molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to get bond indices from.
    reduce_symmetry (bool): If True, then index tuples that can be obtained by invariant permutations are removed. Default is True.

    Returns:
    np.ndarray: A numpy array containing the indices of bonds in the molecule.
    """
    bond_indices = construct_bonds(mol)
    return np.array(list(bond_indices)).astype(np.int64)


def angle_indices(mol) -> np.ndarray:
    """
    Obtain the indices of angles in the given molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to get angle indices from.
    reduce_symmetry (bool): If True, then index tuples that can be obtained by invariant permutations are removed. Default is True.

    Returns:
    np.ndarray: A numpy array containing the indices of angles in the molecule.
    """
    angle_indices = construct_angles(mol)
    return np.array(list(angle_indices)).astype(np.int64)


def torsion_indices(mol) -> np.ndarray:
    """
    Obtain the indices of proper torsions in the given molecule.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to get torsion indices from.
    reduce_symmetry (bool): If True, then index tuples that can be obtained by invariant permutations are removed. Default is True.

    Returns:
    Tuple[np.ndarray, np.ndarray]: Two numpy arrays containing the indices of proper torsions in the molecule.
    """
    propers = construct_propers(mol)
    return np.array(list(propers)).astype(np.int64)



############################################################################################################


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
