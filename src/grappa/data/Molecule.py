"""
Contains the grappa input dataclass 'Molecule'.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from grappa import constants
from grappa.utils import tuple_indices

import pkgutil


@dataclass
class Molecule():
    """
    Input dataclass for grappa parameter description
    Additional features are a dict with name: array/list of shape n_atoms x feat_dim
    """
    # id tuples:
    atoms: Union[list[int], np.ndarray]
    bonds: Union[list[tuple[int,int]], np.ndarray]
    impropers: Union[list[tuple[int,int,int,int]], np.ndarray]

    # atom properties:
    atomic_numbers: list[int]
    partial_charges: list[float]
    additional_features: Dict[str,list] = None
    
    # more id tuples:
    angles: Optional[Union[list[tuple[int,int,int]], np.ndarray]] = None
    propers: Optional[Union[list[tuple[int,int,int,int]], np.ndarray]] = None

    neighbor_dict: Optional[Dict[int, List[int]]] = None # if needed, this is calculated from bonds and stored. This is used to calculate angles and propers if they are not given and later on to check whether torsions are improper.

    def _validate(self):
        # check the input for consistency
        # TODO: compare to current input validation
        # this could check that no equivalent improper torsions or bonds are present
        pass
    
    def  __post_init__(self):
        #TODO: check whether this does what is is supposed to do
        if self.angles is None or self.propers is None:
            tuple_dict = tuple_indices(bonds=self.bonds)
            if self.angles is None:
                self.angles = tuple_dict['angles']
            if self.propers is None:
                self.propers = tuple_dict['propers']

        if self.additional_features is None:
            self.additional_features= {}
 
        # NOTE: add additional, related versions of improper torsions (if not present) to fulfill the required permutation invariance!

        self._validate()
    
    def sort(self):
        """
        Sort the interation tuples to the convention tuple[0] < tuple[-1] using invariant permutations.
        Impropers are not affected.
        """
        for i, bond in enumerate(self.bonds):
            # this notation is agnostic of the data type of self.bonds:
            self.bonds[i] = (bond[0], bond[1]) if bond[0] < bond[1] else (bond[1], bond[0])
        
        for i, angle in enumerate(self.angles):
            self.angles[i] = (angle[0], angle[1], angle[2]) if angle[0] < angle[2] else (angle[2], angle[1], angle[0])

        for i, proper in enumerate(self.propers):
            self.propers[i] = (proper[0], proper[1], proper[2], proper[3]) if proper[0] < proper[3] else (proper[3], proper[2], proper[1], proper[0])

        

    @classmethod
    def from_openmm_system(cls, openmm_system, openmm_topology, improper_central_atom_position:int=None):
        """
        Create a Molecule from an openmm system. If bonds is None, the bonds are extracted from the HarmonicBondForce of the system. For improper torsions, those of the openmm system are used.
        improper_central_atom_position: the position of the central atom in the improper torsions. Defaults to 2, i.e. the third atom in the tuple, which is the amber convention.
        The indices are those of the topology.
        """
        assert pkgutil.find_loader("openmm") is not None, "openmm must be installed to use this constructor."

        import openmm.unit as openmm_unit
        from openmm import System
        from openmm.app import Topology as OpenMMTopology

        assert isinstance(openmm_system, System), f"openmm_system must be an instance of openmm.app.System. but is: {type(openmm_system)}"
        assert isinstance(openmm_topology, OpenMMTopology), f"openmm_topology must be an instance of openmm.app.Topology. but is: {type(openmm_topology)}"

        bonds = []
        for bond in openmm_topology.bonds():
            bonds.append((bond[0].index, bond[1].index))

        neighbor_dict = tuple_indices.get_neighbor_dict(bonds, sort=True)
        tuple_dict = tuple_indices.get_idx_tuples(bonds=bonds, is_sorted=True, neighbor_dict=neighbor_dict)
        angles = tuple_dict['angles']
        propers = tuple_dict['propers']
                
                
        # get the improper torsions from the openmm system:
        impropers = []
        improper_sets = []
        for force in openmm_system.getForces():
            if force.__class__.__name__ == 'PeriodicTorsionForce':
                for i in range(force.getNumTorsions()):
                    torsion = force.getTorsionParameters(i)[0], force.getTorsionParameters(i)[1], force.getTorsionParameters(i)[2], force.getTorsionParameters(i)[3]

                    if set(torsion) in improper_sets:
                        # skip this torsion if it is already present (in a different order, which is fine becasue we always store all three independent orderings for a given central atom and because the central atom is unique assuming there are no fully connected sets of four atoms):
                        continue

                    is_improper, central_idx = tuple_indices.is_improper(ids=torsion, neighbor_dict=neighbor_dict, central_atom_position=improper_central_atom_position)
                    if is_improper:
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
                            if position == constants.IMPROPER_CENTRAL_IDX:
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

                        # also append the set of atoms to the list of improper sets:
                        improper_sets.append(set(torsion))



        # get partial charges from the openmm system:
        partial_charges = []
        for force in openmm_system.getForces():
            if force.__class__.__name__ == 'NonbondedForce':
                for i in range(force.getNumParticles()):
                    q, _, _ = force.getParticleParameters(i)
                    partial_charges.append(q.value_in_unit(openmm_unit.elementary_charge))

        # get atomic numbers and atom indices using zip:
        atomic_numbers, atoms = [], []
        for atom in openmm_topology.atoms():
            atomic_numbers.append(atom.element.atomic_number)
            atoms.append(atom.index)


        return cls(atoms=atoms, bonds=bonds, angles=angles, propers=propers, impropers=impropers, atomic_numbers=atomic_numbers, partial_charges=partial_charges)


    def add_features(self, feat_names:list[str]=['ring_encoding'], **kwargs):
        """
        Add features to the molecule by keyword. Currently supported:
            - 'ring_encoding': a one-hot encoding of ring membership obtained from rdkit. feat dim: 7
            - 'sp_hybridization': a one-hot encoding of the hybridization of the atom. openff_mol must be passed as a keyword argument. feat dim: 6
            - 'is_radical': a boolean indicating whether the atom is a radical or not. feat dim: 1
        """
        for feat_name in feat_names:


            if feat_name == 'ring_encoding':
                assert pkgutil.find_loader("rdkit") is not None, f"rdkit must be installed to use the feature {feat_name}"
                from grappa.utils import rdkit_utils
                
                # translate between atom ids and indices:
                # atom_idx[atom_id] = idx
                atom_idx = {id:idx for idx,id in enumerate(self.atoms)}
                # transform bonds to indices:
                bonds_by_idx = [(atom_idx[bond[0]], atom_idx[bond[1]]) for bond in self.bonds]
                mol = rdkit_utils.rdkit_graph_from_bonds(bonds=bonds_by_idx)
                ring_encoding = rdkit_utils.get_ring_encoding(mol)
                self.additional_features['ring_encoding'] = ring_encoding


            elif feat_name == 'sp_hybridization':
                assert "openff_mol" in kwargs, f"openff_mol must be passed as a keyword argument to use the feature {feat_name}"
                assert pkgutil.find_loader("openff.toolkit") is not None, f"openff.toolkit must be installed to use the feature {feat_name}"
                assert pkgutil.find_loader("rdkit") is not None, f"rdkit must be installed to use the feature {feat_name}"
                from grappa.utils import openff_utils
                openff_mol = kwargs['openff_mol']
                sp_hybridization = openff_utils.get_sp_hybridization_encoding(openff_mol)
                self.additional_features['sp_hybridisation'] = sp_hybridization


            elif feat_name == 'is_radical':
                raise NotImplementedError(f"Feature {feat_name} not implemented yet.")
            

            else:
                raise NotImplementedError(f"Feature {feat_name} not implemented yet.")
    
    def to_dgl(self, max_element=53, exclude_feats:list[str]=[]):
        """
        Converts the molecule to a dgl graph with node features. The elements are one-hot encoded.
        The dgl graph has the following node types:
            - g: global
            - n1: atoms
            - n2: bonds
            - n3: angles
            - n4: propers
            - n4_improper: impropers
        The node type n1 carries the feature 'ids', which are the identifiers in self.atoms. The other interaction levels (n{>1}) carry the idxs (not ids) of the atoms as ordered in self.atoms as feature 'idxs'. These are not the identifiers but must be translated back to the identifiers using ids = self.atoms[idxs] after the forward pass.
        """
        pass


    def to_dict(self):
        """
        Save the molecule as a dictionary of arrays. The additional features are saved as additional dictionary entries. The keys of the additional features must not be the same as the molecule attributes and their values must be numpy arrays.
        """
        assert not any([key in
            [
                'atoms',
                'bonds',
                'angles',
                'propers',
                'impropers',
                'atomic_numbers',
                'partial_charges',
            ]
            for key in self.additional_features.keys()]), f"Additional features must not have the same keys as the molecule attributes but found {self.additional_features.keys()}"
        
        assert all([isinstance(feat, np.ndarray) for feat in self.additional_features.values()]), f"Additional features must be numpy arrays but found {[type(f) for f in self.additional_features.values()]}"

        array_dict = {
            'atoms':np.array(self.atoms).astype(np.int64),
            'bonds':np.array(self.bonds).astype(np.int64),
            'angles':np.array(self.angles).astype(np.int64),
            'propers':np.array(self.propers).astype(np.int64),
            'impropers':np.array(self.impropers).astype(np.int64),
            'atomic_numbers':np.array(self.atomic_numbers).astype(np.int64),
            'partial_charges':np.array(self.partial_charges).astype(np.float32),
            **self.additional_features,
        }
        return array_dict
    

    @classmethod
    def from_dict(cls, array_dict:Dict):
        """
        Create a Molecule from a dictionary of arrays. The additional features are saved as additional dictionary entries. The keys of the additional features must not be the same as the molecule attributes and their values must be numpy arrays.
        """

        assert all([isinstance(feat, np.ndarray) for feat in array_dict.values()]), f"Dict values must be numpy arrays but found {[type(f) for f in array_dict.values()]}"

        return cls(
            atoms=array_dict['atoms'],
            bonds=array_dict['bonds'],
            angles=array_dict['angles'],
            propers=array_dict['propers'],
            impropers=array_dict['impropers'],
            atomic_numbers=array_dict['atomic_numbers'],
            partial_charges=array_dict['partial_charges'],
            additional_features={key:array_dict[key] for key in array_dict.keys() if key not in ['atoms', 'bonds', 'angles', 'propers', 'impropers', 'atomic_numbers', 'partial_charges']}
        )
    
    def save(self, path:str):
        """
        Save the molecule to a npz file.
        """
        np.savez(path, **self.to_dict())

    @classmethod
    def load(cls, path:str):
        """
        Load the molecule from a npz file.
        """
        array_dict = np.load(path)
        return cls.from_dict(array_dict)


    def is_improper(self, torsion):
        """
        Returns is_improper, actual_central_atom_position. See utils/tuple_indices.py for details.
        """
        if self.neighbor_dict is None:
            self.neighbor_dict = tuple_indices.get_neighbor_dict(bonds=self.bonds, sort=True)

        return tuple_indices.is_improper(ids=torsion, neighbor_dict=self.neighbor_dict, central_atom_position=None)