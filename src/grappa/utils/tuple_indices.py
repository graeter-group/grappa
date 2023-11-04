
import numpy as np

from typing import Tuple, Set, Dict, Union, List


def get_idx_tuples(bonds:List[Tuple[int, int]], neighbor_dict:Dict=None, is_sorted:bool=False)->Dict[str, List[Tuple[int, ...]]]:
    """
    This method will return a dictionary with the keys 'bonds', 'angles' and 'propers'.
    The values are lists of tuples, where each tuple contains the indices of the atoms involved in the bond, angle or proper torsion.
    Equivalent tuples are excluded, we sort such that tuple[0] < tuple[-1].
    If a neighbor_dict is provided, we use that to construct the angles and propers, otherwise we construct it ourselves.
    If the is_sorted flag is set to True, we assume that the neighbor lists are sorted, and we do not sort it again. We also assume that the bonds are sorted.
    """

    # generate neighbor_dict dict such that neighbor_dict[atom_id] = [neighbor1, neighbor2, ...]
    if neighbor_dict is None:
        neighbor_dict = get_neighbor_dict(bonds, sort=True)
    else:
        if not is_sorted:
            for atom_id, neighbor_list in neighbor_dict.items():
                neighbor_dict[atom_id] = sorted(neighbor_list)
    # every neighbor list is sorted in ascending order

    angles = []
    propers = []

    for atom1, atom1_neighbors in neighbor_dict.items():

        for i, atom2 in enumerate(atom1_neighbors):
            for atom3 in neighbor_dict[atom2]:
                if atom1 == atom3:
                    continue

                if atom1 < atom3:
                    # assert atom1 < atom3 because of symmetry (abc) = (cba)
                    angles.append((atom1, atom2, atom3)) 

                # now for propers:                
                # divide out the permutation symmetry (abcd) = (dcba)
                for atom4 in neighbor_dict[atom3]:
                    # since the dictionary is sorted, we can enforce atom4 < atom1 by this:
                    if atom4 >= atom1:
                        break # out of the loop over atom4

                    if atom4 == atom2:
                        continue

                    propers.append((atom4, atom3, atom2, atom1)) # this way we ensure that proper[0] < proper[3]


    if not is_sorted:
        bonds = np.sort(np.array(bonds), axis=1)
        bonds = [tuple(bond) for bond in bonds]


    d = {
        'bonds': bonds,
        'angles': angles,
        'propers': propers,
    }

    return d


def get_neighbor_dict(bonds:List[Tuple[int, int]], sort:bool=True)->Dict:
    # generate neighbor_dict dict such that neighbor_dict[atom_id] = [neighbor1, neighbor2, ...]
    neighbor_dict = {}
    for bond in bonds:
        assert len(bond) == 2, f"Encountered bond with more than two atoms: {bond}"
        for i,atom_id in enumerate(bond):
            if neighbor_dict.get(atom_id):
                assert bond[1-i] != atom_id, f"Encountered self-bond: {bond}"
                neighbor_dict[atom_id].append(bond[1-i])
            else:
                neighbor_dict[atom_id] = [bond[1-i]]

    # sort the neighbor_dict:
    if sort:
        for atom_id, neighbor_list in neighbor_dict.items():
            neighbor_dict[atom_id] = sorted(neighbor_list)

    return neighbor_dict



def is_improper(ids:Tuple[int,int,int,int], neighbor_dict:Dict, central_atom_position:int=None)->bool:
    # NOTE: this should also return the central atom position and this positions should be consistent in the Molecule class.
    """
    Returns is_improper, actual_central_atom_position.
    Helper function to check whether the given tuple of indices describes an improper torsion.
    Checks whether the given tuple of indices describes an improper torsion.
    Also the index of the central atom in the input tuple is returned.
    It is assumed that the tuples describe either a proper or improper torsion.
    The central_atom_position is the index of the central atom in the tuple. If known, this can offer an evaluation speedup. This is a convention. In amber force fields, the central atom is the third atom in the tuple, i.e. at position 2.
    """
    # check whether the central atom is the connected to all other atoms in the rdkit molecule.

    if not central_atom_position is None:

        central_atom = ids[central_atom_position]

        # get the neighbor_dict of the central atom
        neighbor_idxs = neighbor_dict[central_atom]

        # for each atom in the torsion, check if it's a neighbor of the central atom
        for i, atom_id in enumerate(ids):
            if i != central_atom_position:  # skip the central atom itself
                if atom_id not in neighbor_idxs:
                    # if one of the atoms is not connected to it, return False
                    return False, None

        # if all atoms are connected to the central atom, this is an improper torsion
        return True, central_atom_position
    
    else:
        # try all atoms as potential central atom:
        for central_atom in ids:
            # get the neighbor_dict of the central atom
            neighbor_idxs = neighbor_dict[central_atom]

            # for each atom in the torsion, check if it's a neighbor of the central atom
            if all([atom_id in neighbor_idxs for atom_id in ids if atom_id != central_atom]):
                return True, ids.index(central_atom)
        
        # we have not found a central atom.
        return False, None