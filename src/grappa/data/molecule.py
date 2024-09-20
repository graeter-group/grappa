"""
Contains the grappa input dataclass 'Molecule'.
"""

from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from grappa import constants
from grappa.utils import tuple_indices
import dgl.heterograph
import torch
from pathlib import Path
import json
import importlib


class Molecule():
    """
    A class representing a molecular graph.

    This class stores the molecular graph and its subgraphs (ie atoms, bonds, angles, torsions), and additional features such as partial charges and ring membership.

    Attributes:
        atoms (np.ndarray[int] of shape (n_atoms)): A list or array of atom identifiers. These identifiers are unique but 
            not necessarily ordered or starting from zero.
        bonds (np.ndarray[int] of shape (n_bonds, 2)): A list of tuples or an np.ndarray
            containing two atom ids representing a bond between them. Each bond may only appear once.
            By convention, the first atom id should be smaller than the second atom id. (This is currently not enforced.)
        impropers (np.ndarray[int] of shape (n_impropers, 4)): A list of tuples or
            an np.ndarray containing four atom ids representing an improper torsion between them.
            Upon construction, one can pass the improper torsions in any order and any amount of permuted versions, the post_init function will sort them, create the three independent versions and store them in the order given by the central atom position.
            After construction, the central atom is always at position grappa.constants.IMPROPER_CENTRAL_IDX.
            Each set of atom ids appears three times for the three independent dihedral angles
            that can be defined for this set of atoms under the constraint that the central atom is given.
            (There are 6==3! possible permutations, but only 3 are independent because of the antisymmetry of the dihedral angle under exchange of the first and last or the second and third atom.)
        atomic_numbers (List[int]): A list of atomic numbers corresponding to the element of each atom in the molecule.
        partial_charges (List[float]): A list of partial charges for each atom in units of the elementary charge.
        additional_features (Optional[Dict[str, List]]): A dictionary containing additional features associated with 
            atoms. The dictionary keys are feature names, and values are lists or arrays of shape (n_atoms, feat_dim).

    Optional Attributes:
        angles (Optional[Union[List[Tuple[int, int, int]], np.ndarray]]): A list or array of tuples, each representing 
            an angle between three atoms. By convention, the first atom ID in the tuple is smaller than the third. 
            If not provided, angles can be calculated from bonds.
        propers (Optional[Union[List[Tuple[int, int, int, int]], np.ndarray]]): A list or array of tuples, each 
            representing a proper torsion between four atoms. By convention, the first atom ID in the tuple is smaller 
            than the fourth. If not provided, proper torsions can be calculated from bonds.
        neighbor_dict (Optional[Dict[int, List[int]]]): An internal variable storing neighbor relationships of atoms. 
            It is calculated from bonds and used for deriving angles, proper and improper torsions if they are not 
            explicitly provided.

    Note:
        - The `additional_features` attribute can be used to add custom features relevant to specific molecule types.
    """

    def __init__(
        self,
        atoms: Union[List[int], np.ndarray],
        bonds: Union[List[Tuple[int, int]], np.ndarray],
        impropers: Union[List[Tuple[int, int, int, int]], np.ndarray],
        atomic_numbers: List[int],
        partial_charges: List[float],
        additional_features: Optional[Dict[str, List]] = None,
        angles: Optional[Union[List[Tuple[int, int, int]], np.ndarray]] = None,
        propers: Optional[Union[List[Tuple[int, int, int, int]], np.ndarray]] = None,
        improper_in_correct_format: bool = False,
        ring_encoding: bool = True,
        degree: bool = True,
        mass_encoding: bool = True,
        mapped_smiles: str = None,
        charge_model: str = 'None',
    )->None:
        self.atoms = atoms if isinstance(atoms, list) else atoms.tolist()
        self.bonds = bonds if isinstance(bonds, list) else bonds.tolist()
        self.impropers = impropers if isinstance(impropers, list) else impropers.tolist()
        self.atomic_numbers = atomic_numbers if isinstance(atomic_numbers, list) else atomic_numbers.tolist()
        self.partial_charges = partial_charges if isinstance(partial_charges, list) else partial_charges.tolist()
        self.additional_features = additional_features
        self.angles = angles
        self.propers = propers
        self.neighbor_dict = None  # Calculated from bonds if needed. key: atom id, value: list of neighbor atom ids

        self.charge_model = charge_model

        if not improper_in_correct_format:
            self.process_impropers()

        self.__post_init__()
        self._validate()


        if mass_encoding:
            self.add_features(feat_names=['mass'])

        if ring_encoding:
            self.add_features(feat_names=['ring_encoding'])

        if degree:
            self.add_features(feat_names=['degree'])

    def process_impropers(self):
        """
        Updates the impropers such that they are as described in the class docstring.
        """
        if self.neighbor_dict is None:
            self.neighbor_dict = tuple_indices.get_neighbor_dict(bonds=self.bonds, sort=True)
            
        _, self.impropers = tuple_indices.get_torsions(torsion_ids=self.impropers, neighbor_dict=self.neighbor_dict, central_atom_position=constants.IMPROPER_CENTRAL_IDX)


    def _validate(self):
        # check the input for consistency
        # TODO: compare to current input validation
        pass
    
    
    def  __post_init__(self):
        if self.angles is None or self.propers is None:
            is_sorted = False

            if self.neighbor_dict is None:
                self.neighbor_dict = tuple_indices.get_neighbor_dict(bonds=self.bonds, sort=True)
                is_sorted = True

            tuple_dict = tuple_indices.get_idx_tuples(bonds=self.bonds, neighbor_dict=self.neighbor_dict, is_sorted=is_sorted)

            if self.angles is None:
                self.angles = tuple_dict['angles'] if isinstance(tuple_dict['angles'], list) else tuple_dict['angles'].tolist()
            if self.propers is None:
                self.propers = tuple_dict['propers'] if isinstance(tuple_dict['propers'], list) else tuple_dict['propers'].tolist()

        if self.additional_features is None:
            self.additional_features= {}

        if not self.charge_model in constants.CHARGE_MODELS:
            raise ValueError(f"charge_model must be one of {constants.CHARGE_MODELS} but is {self.charge_model}")
        
        if not 'charge_model' in self.additional_features.keys():
            assert self.charge_model in constants.CHARGE_MODELS, f"charge_model must be one of {constants.CHARGE_MODELS} but is {self.charge_model}"
            assert len(constants.CHARGE_MODELS) <= constants.MAX_NUM_CHARGE_MODELS, f"the number of charge models must be less than or equal to {constants.MAX_NUM_CHARGE_MODELS}"
            
            ALL_CHARGE_MODELS = constants.CHARGE_MODELS + ['None'] * (constants.MAX_NUM_CHARGE_MODELS - len(constants.CHARGE_MODELS)) # this is done to ensure that the number of charge models can be modified without making past model weights incompatible.

            self.additional_features['charge_model'] = np.tile(np.array([cm == self.charge_model for cm in ALL_CHARGE_MODELS], dtype=np.float32), (len(self.atoms),1))


        # initialize all mols to be not-radical if not overwritten later:
        if not 'is_radical' in self.additional_features.keys():
            self.additional_features['is_radical'] = np.zeros((len(self.atoms),), dtype=np.float32)

        # ensure that angles and propers are lists:
        self.angles = self.angles if isinstance(self.angles, list) else self.angles.tolist()
        self.propers = self.propers if isinstance(self.propers, list) else self.propers.tolist()

        self._validate()

    
    
    def sort(self):
        """
        Sort the interation tuples to the convention tuple[0] < tuple[-1] using invariant permutations.
        Impropers are not affected.
        For propers, this is not unique, since permutation of outer atoms is also allowed if the phase is zero. We do not apply this permutation here.
        """
        for i, bond in enumerate(self.bonds):
            # this notation is agnostic of the data type of self.bonds:
            self.bonds[i] = (bond[0], bond[1]) if bond[0] < bond[1] else (bond[1], bond[0])
        
        for i, angle in enumerate(self.angles):
            self.angles[i] = (angle[0], angle[1], angle[2]) if angle[0] < angle[2] else (angle[2], angle[1], angle[0])

        for i, proper in enumerate(self.propers):
            self.propers[i] = (proper[0], proper[1], proper[2], proper[3]) if proper[0] < proper[3] else (proper[3], proper[2], proper[1], proper[0])


    @classmethod
    def from_openmm_system(cls, openmm_system, openmm_topology, partial_charges:Union[list,float,np.ndarray]=None, ring_encoding:bool=True, mapped_smiles:str=None, charge_model:str='None'):
        """
        Create a Molecule from an openmm system. The bonds are extracted from the HarmonicBondForce of the system. For improper torsions, those of the openmm system are used.
        improper_central_atom_position: the position of the central atom in the improper torsions. Defaults to 2, i.e. the third atom in the tuple, which is the amber convention.
        The indices are those of the topology.
        the topology may be a sub-topology of the full system, e.g. without solvant. The indices have to correspond to the indices of the atoms in the system.
        Arguments:
            - openmm_system: an openmm system
            - openmm_topology: an openmm topology
            - partial_charges: a list of partial charges for each atom in units of the elementary charge. If None, the partial charges are obtained from the openmm system.
            - ring_encoding: if False, the ring encoding feature (for which rdkit is needd) is not added.
            - mapped_smiles: the mapped smiles string of the molecule. If not None, this information is used to initialize the additional feature 'sp_hybridization'.

            
        Args:
            openmm_system ([openmm.System]): an openmm system defining the improper torsions and, if partial_charges is None, the partial charges.
            openmm_topology ([openmm.app.Topology]): an openmm topology defining the bonds and atomic numbers.
            partial_charges ([type], optional): a list of partial charges for each atom in units of the elementary charge. If None, the partial charges are obtained from the openmm system. Defaults to None.
            ring_encoding (bool, optional): if True, the ring encoding feature (for which rdkit is needd) is added. Defaults to True.
            mapped_smiles (str, optional): the mapped smiles string of the molecule. If not None, this information is used to initialize the additional feature 'sp_hybridization'. Defaults to None.
            """
        assert importlib.util.find_spec("openmm") is not None, "openmm must be installed to use this constructor."

        import openmm.unit as openmm_unit
        from openmm import System
        from openmm.app import Topology as OpenMMTopology

        assert isinstance(openmm_system, System), f"openmm_system must be an instance of openmm.app.System. but is: {type(openmm_system)}"
        assert isinstance(openmm_topology, OpenMMTopology), f"openmm_topology must be an instance of openmm.app.Topology. but is: {type(openmm_topology)}"

        # indices in the system:
        if openmm_system.getNumParticles() > len(list(openmm_topology.atoms())):
            atom_idxs = [int(atom.id) for atom in openmm_topology.atoms()] # assume that the id in the topology is the index in the system.
        elif openmm_system.getNumParticles() == len(list(openmm_topology.atoms())):
            atom_idxs = list(range(openmm_system.getNumParticles()))
        else:
            raise ValueError(f"the number of particles in the system ({openmm_system.getNumParticles()}) must be equal to or greater than the number of atoms in the topology ({len(list(openmm_topology.atoms()))})")
            
        bonds = []
        for bond in openmm_topology.bonds():
            bonds.append((bond[0].index, bond[1].index)) # here we use the index in atom_idxs

        neighbor_dict = tuple_indices.get_neighbor_dict(bonds, sort=True)
        tuple_dict = tuple_indices.get_idx_tuples(bonds=bonds, is_sorted=True, neighbor_dict=neighbor_dict)
        angles = tuple_dict['angles']
        propers = tuple_dict['propers']

        # get the improper torsions:
        all_torsions = []
        for force in openmm_system.getForces():
            if force.__class__.__name__ == 'PeriodicTorsionForce':
                for i in range(force.getNumTorsions()):
                    *torsion, _,_,_ = force.getTorsionParameters(i)
                    assert len(torsion) == 4, f"torsion must have length 4 but has length {len(torsion)}"

                    # add the torsion if it is between atoms included in the topology:
                    if all([atom_idx in atom_idxs for atom_idx in torsion]):
                        all_torsions.append(tuple(torsion))

        _, impropers = tuple_indices.get_torsions(all_torsions, neighbor_dict=neighbor_dict, central_atom_position=constants.IMPROPER_CENTRAL_IDX)

        if partial_charges is None:
            # get partial charges from the openmm system:
            partial_charges = []
            for force in openmm_system.getForces():
                if force.__class__.__name__ == 'NonbondedForce':
                    for i in atom_idxs:
                        q, _, _ = force.getParticleParameters(i)
                        partial_charges.append(q.value_in_unit(openmm_unit.elementary_charge))

        elif isinstance(partial_charges, int):
            partial_charges = [partial_charges] * len(list(openmm_topology.atoms()))
        elif isinstance(partial_charges, np.ndarray):
            partial_charges = partial_charges.tolist()
        else:
            if not isinstance(partial_charges, list):
                raise ValueError(f"partial_charges must be None, int or np.ndarray but is {type(partial_charges)}")

        # get atomic numbers (order is the same as atom_idxs)
        atomic_numbers = []
        for atom in openmm_topology.atoms():
            atomic_numbers.append(atom.element.atomic_number)

        self = cls(atoms=atom_idxs, bonds=bonds, angles=angles, propers=propers, impropers=impropers, atomic_numbers=atomic_numbers, partial_charges=partial_charges, improper_in_correct_format=True, ring_encoding=ring_encoding, mapped_smiles=mapped_smiles, degree=True, charge_model=charge_model)


        return self


    def add_features(self, feat_names:Union[str,List[str]]=['ring_encoding', 'degree', 'mass'], **kwargs):
        """
        Add features to the molecule by keyword. Currently supported:
            - 'ring_encoding': a one-hot encoding of membership in rings of size 3 to 8. feat dim: 7
            - 'sp_hybridization': a one-hot encoding of the hybridization of the atom. openff_mol must be passed as a keyword argument. feat dim: 6
            - 'is_aromatic': a one-hot encoding indicating whether the atom is aromatic or not. openff_mol must be passed as a keyword argument. feat dim: 1
            - 'is_radical': a one-hot encoding indicating whether the atom is a radical or not. feat dim: 1
            - 'degree': the degree of the node in the graph, i.e. the number of neighbors, one hot encoded. feat dim: 6
            - 'mass': the mass and the nat log of the mass of the atom. openff_mol must be passed as a keyword argument. feat dim: 2
            - 'partial_charge_encoding': 
        """
        if isinstance(feat_names, str):
            feat_names = [feat_names]

        for feat_name in feat_names:

            if feat_name == 'ring_encoding':
                assert importlib.util.find_spec("rdkit") is not None, f"rdkit must be installed to use the feature {feat_name}"
                from grappa.utils import rdkit_utils
                
                # translate between atom ids and indices:
                # atom_idx[atom_id] = idx
                atom_idx = {id:idx for idx,id in enumerate(self.atoms)}
                # transform bonds to indices:
                bonds_by_idx = [(atom_idx[bond[0]], atom_idx[bond[1]]) for bond in self.bonds]
                mol = rdkit_utils.rdkit_graph_from_bonds(bonds=bonds_by_idx)
                ring_encoding_ = rdkit_utils.get_ring_encoding(mol)
                self.additional_features['ring_encoding'] = ring_encoding_

            elif feat_name == 'degree':
                assert importlib.util.find_spec("rdkit") is not None, f"rdkit must be installed to use the feature {feat_name}"
                from grappa.utils import rdkit_utils

                # translate between atom ids and indices:
                # atom_idx[atom_id] = idx
                atom_idx = {id:idx for idx,id in enumerate(self.atoms)}
                # transform bonds to indices:
                bonds_by_idx = [(atom_idx[bond[0]], atom_idx[bond[1]]) for bond in self.bonds]
                mol = rdkit_utils.rdkit_graph_from_bonds(bonds=bonds_by_idx)

                degree_ = rdkit_utils.get_degree(mol)
                self.additional_features[feat_name] = degree_


            elif feat_name == 'mass':
                masses = np.array([constants.ATOMIC_MASSES[atomic_number] for atomic_number in self.atomic_numbers], dtype=np.float32)
                masses_log = np.log(masses)
                self.additional_features[feat_name] = np.stack((masses, masses_log), axis=1)


            elif feat_name == 'sp_hybridization':
                assert "openff_mol" in kwargs, f"openff_mol must be passed as a keyword argument to use the feature {feat_name}"
                assert importlib.util.find_spec("openff.toolkit") is not None, f"openff.toolkit must be installed to use the feature {feat_name}"
                assert importlib.util.find_spec("rdkit") is not None, f"rdkit must be installed to use the feature {feat_name}"
                from grappa.utils import openff_utils
                openff_mol = kwargs['openff_mol']
                sp_hybridization = openff_utils.get_sp_hybridization_encoding(openff_mol)
                self.additional_features[feat_name] = sp_hybridization




            elif feat_name == 'is_aromatic':
                assert "openff_mol" in kwargs, f"openff_mol must be passed as a keyword argument to use the feature {feat_name}"
                assert importlib.util.find_spec("openff.toolkit") is not None, f"openff.toolkit must be installed to use the feature {feat_name}"
                assert importlib.util.find_spec("rdkit") is not None, f"rdkit must be installed to use the feature {feat_name}"
                from grappa.utils import openff_utils
                openff_mol = kwargs['openff_mol']
                is_aromatic = openff_utils.get_is_aromatic(openff_mol)
                self.additional_features[feat_name] = is_aromatic


            elif feat_name == 'is_radical':
                raise NotImplementedError(f"Feature {feat_name} not implemented yet.")

            else:
                raise NotImplementedError(f"Feature {feat_name} not implemented yet.")
            

    @classmethod
    def from_smiles(cls, mapped_smiles:str, openff_forcefield:str='openff-1.2.0.offxml', partial_charges:Union[np.ndarray, int]=None, charge_model:str='None'):
        """
        DEPRECATED, USE from_openff_molecule INSTEAD.
        Create a Molecule from a mapped smiles string and an openff forcefield. The openff_forcefield is used to obtain improper torsions and, if partial_charges is None, to obtain the partial charges.
        """
        assert importlib.util.find_spec("openff.toolkit") is not None, "openff.toolkit must be installed to use this constructor."

        from grappa.utils import openff_utils

        # get the openmm system and topology:
        system, topol, openff_mol = openff_utils.get_openmm_system(mapped_smiles=mapped_smiles, openff_forcefield=openff_forcefield, partial_charges=partial_charges)

        # get the molecule from the openmm system and topology:
        mol = cls.from_openmm_system(openmm_system=system, openmm_topology=topol, partial_charges=partial_charges, charge_model=charge_model)

        # add features:
        mol.add_features(feat_names=['ring_encoding', 'sp_hybridization', 'is_aromatic'], openff_mol=openff_mol)

        return mol


    @classmethod
    def from_openff_molecule(cls, openff_mol, partial_charges:Union[np.ndarray, float, List[float]]=None, impropers:Union[str, List[Tuple[int,int,int,int]]]='smirnoff', charge_model:str='None'):
        """
        Creates a Molecule from an openff molecule. The openff molecule must have partial charges id partial_charges is None.
        impropers can either be a method, 'smirnoff' or 'amber', or a list of tuples of atom ids.
        Args:
            openff_mol (openff.toolkit.topology.Molecule): an openff molecule
            partial_charges (Union[np.ndarray, float, List[float]]): the partial charges of the molecule. If None, the partial charges are obtained from the openff molecule.
            impropers (Union[str, List[Tuple[int,int,int,int]]]): the improper torsions of the molecule. If 'smirnoff' or 'amber', the improper torsions are obtained from the openff molecule. If a list of tuples of atom ids, these are used as the improper torsions.
        """
        assert importlib.util.find_spec("openff.toolkit") is not None, "openff.toolkit must be installed to use this constructor."

        atoms = [atom.molecule_atom_index for atom in openff_mol.atoms]

        atomic_numbers = [atom.atomic_number for atom in openff_mol.atoms]

        # get a list of bond idxs where the first atom idx is smaller than the second:
        bonds = [(bond.atom1_index, bond.atom2_index) if bond.atom1_index < bond.atom2_index else (bond.atom2_index, bond.atom1_index) for bond in openff_mol.bonds]

        if partial_charges is None:
            assert openff_mol.partial_charges is not None, f"partial_charges must be passed as an argument or be present in the openff molecule but both are None"

            from openff.units import unit
            partial_charges = (openff_mol.partial_charges/unit.elementary_charge).magnitude

        if isinstance(partial_charges, int):
            partial_charges = [partial_charges] * len(atoms)
        elif isinstance(partial_charges, np.ndarray):
            partial_charges = partial_charges.tolist()
        else:
            if not isinstance(partial_charges, list):
                raise ValueError(f"partial_charges must be None, int or np.ndarray but is {type(partial_charges)}")


        # get the impropers:
        if isinstance(impropers, str):
            if impropers == 'smirnoff':
                impropers = openff_mol.smirnoff_impropers

            elif impropers == 'amber':
                impropers = openff_mol.amber_impropers

            # now get the indices of these, but only one version of each improper since we generate the other permutations in the post_init function:
            impropers = list(set(
                tuple(
                    sorted((atoms[0]._molecule_atom_index, atoms[1]._molecule_atom_index, atoms[2]._molecule_atom_index, atoms[3]._molecule_atom_index))
                ) for atoms in impropers
            ))
            
        # initialize with the corresponding flag to covnert the impropers to grappa format:
        mol = cls(atoms=atoms, bonds=bonds, impropers=impropers, atomic_numbers=atomic_numbers, partial_charges=partial_charges, improper_in_correct_format=False, charge_model=charge_model)

        # add features:
        mol.add_features(feat_names=['ring_encoding', 'sp_hybridization', 'is_aromatic'], openff_mol=openff_mol)

        return mol
    
    
    def to_dgl(self, max_element=constants.MAX_ELEMENT, exclude_feats:List[str]=[]):
        """
        Converts the molecule to a dgl graph with node features. The elements are one-hot encoded.
        The dgl graph has the following node types (if the number of respective nodes is nonzero):
            - g: global
            - n1: atoms
            - n2: bonds
            - n3: angles
            - n4: propers
            - n4_improper: impropers        
        The node type n1 carries the feature 'ids', which are the identifiers in self.atoms. The other interaction levels (n{>1}) carry the idxs (not ids) of the atoms as ordered in self.atoms as feature 'idxs'. These are not the identifiers but must be translated back to the identifiers using ids = self.atoms[idxs] after the forward pass.
        atomic numbers are one-hot encoded such that Z=argmax(g.nodes['n1'].data['atomic_number'])+1.
        """
        assert max_element > 0, f"max_element must be larger than 0 but is {max_element}"
        assert not any([ids is None for ids in [self.angles, self.propers]]), f"atoms, bonds, angles, propers and impropers must not be None"
        
        # initialize empty dictionary
        hg = {}

        idx_from_id = {id:idx for idx, id in enumerate(self.atoms)}

        # transform entries of n{>1} to idxs of the atoms:
        idxs = {
            "n1": torch.tensor(self.atoms, dtype=torch.int64), # these are ids
            "n2": torch.tensor([(idx_from_id[bond[0]], idx_from_id[bond[1]]) for bond in self.bonds], dtype=torch.int64), # these are idxs
            "n3": torch.tensor([(idx_from_id[angle[0]], idx_from_id[angle[1]], idx_from_id[angle[2]]) for angle in self.angles], dtype=torch.int64), # these are idxs
            "n4": torch.tensor([(idx_from_id[proper[0]], idx_from_id[proper[1]], idx_from_id[proper[2]], idx_from_id[proper[3]]) for proper in self.propers], dtype=torch.int64), # these are idxs
            "n4_improper": torch.tensor([(idx_from_id[improper[0]], idx_from_id[improper[1]], idx_from_id[improper[2]], idx_from_id[improper[3]]) for improper in self.impropers], dtype=torch.int64), # these are idxs
        }

        # define the heterograph structure:

        b = idxs["n2"].transpose(0,1) # transform from (n_bonds, 2) to (2, n_bonds)

        assert b.max() < len(self.atoms), f"Internal error: Maximal atom index in bonds ({b.max()}) must be smaller than the number of atoms ({len(self.atoms)})"

        # the edges of the other direction:
        first_idxs = torch.cat((b[0], b[1]), dim=0)
        second_idxs = torch.cat((b[1], b[0]), dim=0) # shape (2*n_bonds,)

        # assert that every atom has a bond (completely isolated atoms will not be included in the graph since it is constructed via an adjacency matrix and we have no self-bonds!)
        # use torch unique to check if all atoms are in the bonds:
        assert len(torch.unique(b.flatten())) == len(self.atoms), f"Every atom must be part of a bond but {len(torch.unique(b.flatten()))} of {len(self.atoms)} atoms are in the bonds. Atoms {[self.atoms[idx_] for idx_ in np.setdiff1d(np.arange(len(self.atoms)), np.unique(b.flatten().detach().numpy()))]} are not part of a bond. These might be ions, which are generally not parameterized by Grappa."

        hg[("n1", "n1_edge", "n1")] = torch.stack((first_idxs, second_idxs), dim=0).int() # shape (2, 2*n_bonds)
        assert len(self.bonds)*2 == len(hg[("n1", "n1_edge", "n1")][0]), f"number of edges in graph ({len(hg[('n1', 'n1_edge', 'n1')][0])}) does not match 2*number of bonds ({2*len(self.bonds)})"
            
        # ======================================
        # since we do not need neighborhood for levels other than n1, simply create a graph with only self loops:
        # ======================================
        TERMS = ["n2", "n3", "n4", "n4_improper"]

        for term in TERMS+["g"]:
            key = (term, f"{term}_edge", term)
            n_nodes = len(idxs[term]) if term not in ["g"] else 1
            hg[key] = torch.stack(
                [
                    torch.arange(n_nodes),
                    torch.arange(n_nodes),
                ], dim=0).int()

        # transform to tuples of tensors:
        hg = {key: (value[0], value[1]) for key, value in hg.items()}

        for k, (vsrc,vdest) in hg.items():
            # make sure that the tensors have the correct shape:
            assert vsrc.shape == vdest.shape, f"shape of {k} is {vsrc.shape} and {vdest.shape}"

        # init graph
        hg = dgl.heterograph(hg)

        # write indices in the nodes
        for term in TERMS:
            hg.nodes[term].data["idxs"] = idxs[term]

        # for n1, call the feature 'ids' instead of 'idxs' because these are the identifiers of the atoms, not indices:
        hg.nodes["n1"].data["ids"] = idxs["n1"]

        assert len(self.bonds)*2 == hg.num_edges('n1_edge'), f"number of n1_edges in graph ({hg.num_edges('n1_edge')}) does not match 2 times number of bonds ({len(self.bonds)})"

        # add standard features (atomic number and partial charge):
        if isinstance(self.atomic_numbers, list):
            if max(self.atomic_numbers) > max_element:
                raise ValueError(f"max_element ({max_element}) must be larger than the largest atomic number ({max(self.atomic_numbers)})")
            if min(self.atomic_numbers) < 1:
                raise ValueError(f"min_element must be larger than 0 but is {min(self.atomic_numbers)}")
        else:
            if np.any(self.atomic_numbers > max_element):
                raise ValueError(f"max_element ({max_element}) must be larger than the largest atomic number ({np.max(self.atomic_numbers)})")
            if np.any(self.atomic_numbers < 1):
                raise ValueError(f"min_element must be larger than 0 but is {np.min(self.atomic_numbers)}")

        # we have no atomic number 0, so we can safely subtract 1 from all atomic numbers:
        atom_onehot = torch.nn.functional.one_hot(torch.tensor(self.atomic_numbers)-1, num_classes=max_element).float()

        assert len(self.atomic_numbers) == len(self.partial_charges) == len(self.atoms) == hg.num_nodes('n1'), f"number of atoms ({len(self.atoms)}), atomic numbers ({len(self.atomic_numbers)}), partial charges ({len(self.partial_charges)}) and nodes in n1 ({hg.num_nodes('n1')}) must be equal"

        hg.nodes["n1"].data["atomic_number"] = atom_onehot
        hg.nodes["n1"].data["partial_charge"] = torch.tensor(self.partial_charges, dtype=torch.float32)

        # add additional features:
        for feat in self.additional_features.keys():
            if feat in exclude_feats:
                continue
            try:
                hg.nodes["n1"].data[feat] = torch.tensor(self.additional_features[feat], dtype=torch.float32)
            except Exception as e:
                raise Exception(f"Failed to add feature {feat} to the graph. Error: {e}")

        return hg



    def to_dict(self):
        """
        Save the molecule as a dictionary of arrays. The additional features are saved as additional dictionary entries. The keys of the additional features must not be the same as the molecule attributes and their values must be numpy arrays.
        """
        # check that the additional features are not the same as the molecule attributes since keys must be unique:
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

        additional_features = {key:array_dict[key] for key in array_dict.keys() if key not in ['atoms', 'bonds', 'angles', 'propers', 'impropers', 'atomic_numbers', 'partial_charges']}

        assert all([feat.shape[0] == array_dict['atoms'].shape[0] for feat in additional_features.values()]), f"Additional features must have the same number of atoms as the molecule but found {[feat.shape[0] for feat in additional_features.values()]}"

        return cls(
            atoms=array_dict['atoms'],
            bonds=array_dict['bonds'],
            angles=array_dict['angles'],
            propers=array_dict['propers'],
            impropers=array_dict['impropers'],
            atomic_numbers=array_dict['atomic_numbers'],
            partial_charges=array_dict['partial_charges'],
            additional_features=additional_features,
            improper_in_correct_format=True, # assume this is already in the correct format since it was stored
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
    

    def to_list_dict(self):
        """
        Returns a dictionary of lists describing the molecule. This is just a wrapper for the to_dict method that writes numpy arrays.
        """
        array_dict = self.to_dict()
        list_dict = {key:array_dict[key].tolist() for key in array_dict.keys()}
        return list_dict
    

    def to_json(self, filename:Union[Path,str]):
        with open(filename, 'w') as f:
            json.dump(self.to_list_dict(), f, indent=4)

    @classmethod
    def from_json(cls, filename:Union[Path,str]):
        with open(filename, 'r') as f:
            list_dict = json.load(f)
        return cls.from_list_dict(list_dict)
    

    @classmethod
    def from_list_dict(cls, list_dict:Dict):
        """
        Create a Molecule from a dictionary of lists. This method is just a wrapper for the from_dict method that expects arrays.
        """
        array_dict = {key:np.array(list_dict[key]) for key in list_dict.keys()}
        return cls.from_dict(array_dict)


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        features_str = ', '.join(list(self.additional_features.keys()))
        return f"<grappa.data.Molecule ({len(self.atoms)} atoms, {len(self.bonds)} bonds, {len(self.angles)} angles, {len(self.propers)} propers, {len(self.impropers)//3} impropers, features: {features_str})>"

    def set_radical_flag(self, atom_id:int, is_radical:bool=True):
        """
        Set the radical flag of an atom.
        """
        assert atom_id in self.atoms, f"atom_id {atom_id} not in molecule"
        self.additional_features['is_radical'][self.atoms.index(atom_id)] = 1.0 if is_radical else 0.0

    def set_radical_feature(self, is_radical:Union[List[bool], np.ndarray]):
        """
        Set the radical flag of atoms.
        """
        assert len(is_radical) == len(self.atoms), f"length of is_radical ({len(is_radical)}) must be equal to number of atoms ({len(self.atoms)})"
        if isinstance(is_radical, np.ndarray):
            assert len(is_radical.shape) == 1, f"is_radical must be a 1d array but has shape {is_radical.shape}"

        self.additional_features['is_radical'] = np.array(is_radical, dtype=np.float32)

    @classmethod
    def random(cls):
        """
        Create a random molecule (A-B-C-D, E-B) with atomic numbers 1,2,3,4,5 and partial charges 0.0, 0.2, 0.3, -0.5., 0.
        """

        atoms = [0,1,2,3,4]
        bonds = [(0,1), (1,2), (2,3), (1,4)]
        angles = [(0,1,2), (1,2,3), (1,2,4)]
        propers = [(0,1,2,3)]
        impropers = [(0,2,1,4)]

        atomic_numbers = [1,2,3,4,5]
        partial_charges = [0.0, 0.2, 0.3, -0.5, 0.]

        return cls(atoms=atoms, bonds=bonds, angles=angles, propers=propers, impropers=impropers, atomic_numbers=atomic_numbers, partial_charges=partial_charges)