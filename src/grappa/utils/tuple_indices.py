
import numpy as np
from grappa.constants import IMPROPER_CENTRAL_IDX
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
    Helper function to check whether the given tuple of indices describes an improper torsion, i.e. whether there is one atom that is connected to all other atoms in the tuple. The index of the central atom in the input tuple is returned as actual_central_atom_position. If known, this can offer an evaluation speedup. This is a convention. In amber force fields, the central atom is the third atom in the tuple, i.e. at position 2. It is advised to not rely on conventions but leave this argument as None.
    """
    # check whether the central atom is the connected to all other atoms in the rdkit molecule.

    if isinstance(ids, np.ndarray):
        ids = tuple(ids.tolist())

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
        # order: 2,1,0,3 because position 2 is the central atom in amber force fields (->small speedup)
        for central_atom in (ids[i] for i in [2,1,0,3]):
            # get the neighbor_dict of the central atom
            neighbor_idxs = neighbor_dict[central_atom]

            # for each atom in the torsion, check if it's a neighbor of the central atom
            if all([atom_id in neighbor_idxs for atom_id in ids if atom_id != central_atom]):
                return True, ids.index(central_atom)
        
        # we have not found a central atom.
        return False, None
    
    
def is_proper(ids:Tuple[int,int,int,int], neighbor_dict:Dict)->bool:
    """
    Returns is_proper.
    Helper function to check whether the given tuple of indices describes a proper torsion, i.e. whether ids[0] is connected to ids[1], ids[1] to ids[2] and ids[2] to ids[3].
    is_proper and is_improper are not per se mutually exclusive. This would require, however, a ring of size 4 which is unusual for molecules.
    """
    bond_01 = ids[0] in neighbor_dict[ids[1]]
    bond_12 = ids[1] in neighbor_dict[ids[2]]
    bond_23 = ids[2] in neighbor_dict[ids[3]]

    return bond_01 and bond_12 and bond_23



def get_torsions(torsion_ids:List[Tuple[int,int,int,int]], neighbor_dict:Dict, central_atom_position=IMPROPER_CENTRAL_IDX)->Tuple[List[Tuple[int,int,int,int]], List[Tuple[int,int,int,int]]]:
    """
    Returns propers, impropers in the format required by grappa.data.Molecule.

    torsion_ids is a list of tuples of atom ids, where each tuple contains four atom ids and each tuple describes a proper or improper torsion. For proper torsions, the order of the atoms is important (atoms that are connected to each other must appear consecutively), for improper torsions, it is not.
    
    propers contains the atom ids of proper torsions, where each set of four atom ids only occurs once.

    impropers contains the atom ids of improper torsions with the central position at central_atom_position. Each set of atom ids appears three times for the three independent dihedral angles that can be defined for this set of atoms under the constraint that the central atom is given. (There are 6==3! possible permutations, but only 3 are independent because of the antisymmetry of the dihedral angle under exchange of the first and last or the second and third atom.)
    """
    propers = []
    impropers = []
    improper_set = set([])
    proper_set = set([])

    for torsion in torsion_ids:

        if tuple(sorted(torsion)) in improper_set or tuple(sorted(torsion)) in proper_set:
            # skip this torsion if it is already present (potentially with a different order)
            continue

        torsion_is_improper, central_idx = is_improper(ids=torsion, neighbor_dict=neighbor_dict)
        torsion_is_proper = is_proper(ids=torsion, neighbor_dict=neighbor_dict)

        # NOTE: CHECK THIS
        # if a torsion is both proper and improper, we consider it proper.
        if torsion_is_improper and torsion_is_proper:
            torsion_is_improper = False

        if not torsion_is_proper and not torsion_is_improper:
            raise RuntimeError(f"Encountered torsion that is neither proper nor improper: {torsion}")


        if not torsion_is_improper:
            # append the torsion to the list of propers:
            propers.append(torsion)

            # sort the torsion using an invariant permutation (reversal):
            torsion = torsion if torsion[0] < torsion[-1] else tuple(reversed(torsion))

            # also append the set of atoms to the list of proper sets:
            proper_set.add(tuple(sorted(torsion)))

        if torsion_is_improper:
            # permute two atoms such that the central atom is at the index given by grappa.constants.IMPROPER_CENTRAL_IDX:
            central_atom = torsion[central_idx]
            other_atoms = [torsion[i] for i in range(4) if i != central_idx]
            # now permute the torsion cyclically while keeping the central atom at its position to obtain the other two independent orderings:
            other_atoms2 = [other_atoms[i] for i in (1,2,0)]
            other_atoms3 = [other_atoms[i] for i in (2,0,1)]

            # now form the three versions of the torsion tuple such that the central atom is always at the same position and the other atoms are taken in the order of the respective other_atoms list:
            torsion1, torsion2, torsion3 = [], [], []
            other_atom_position = 0
            for position in range(4):
                if position == central_atom_position:
                    torsion1.append(central_atom)
                    torsion2.append(central_atom)
                    torsion3.append(central_atom)
                else:
                    torsion1.append(other_atoms[other_atom_position])
                    torsion2.append(other_atoms2[other_atom_position])
                    torsion3.append(other_atoms3[other_atom_position])
                    other_atom_position += 1

            # now append the three torsions to the list of impropers:
            impropers.append(tuple(torsion1))
            impropers.append(tuple(torsion2))
            impropers.append(tuple(torsion3))

            # also append the set of atoms
            improper_set.add(tuple(sorted(torsion1)))

    return propers, impropers