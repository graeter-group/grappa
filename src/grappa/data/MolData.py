"""
Contains the grappa input dataclass 'MolData', which is an extension of the dataclass 'Molecule' that contains conformational data and characterizations like smiles string or PDB file.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from grappa.data import Molecule, Parameters

import pkgutil


@dataclass
class MolData():
    """
    Dataclass for entries in datasets on which grappa can be trained.
    """
    molecule: Molecule
    classical_parameters: Parameters # these are used for regularisation and to estimate the statistics of the reference energy and gradient

    # conformational data:
    xyz: np.ndarray
    energy: np.ndarray
    gradient: np.ndarray

    # reference values (centered bonded energy, bonded gradient)
    reference_energy: np.ndarray
    reference_gradient: np.ndarray
    
    # additional characterizations:
    mapped_smiles: Optional[str] = None
    pdb: Optional[str] = None

    # nonbonded contributions:
    nonbonded_energy: Optional[np.ndarray] = None
    nonbonded_gradient: Optional[np.ndarray] = None


    def _validate(self):
        # parameter atoms must be same as molecule atoms
        # check shapes

        pass   
    
    def  __post_init__(self):

        self._validate()
    
    
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
        Also create entries 'u_ref' and 'grad_ref' in the global node type g and in the atom node type n1, which contain the reference energy and gradient, respectively.
        """
        g = self.molecule.to_dgl(max_element=max_element, exclude_feats=exclude_feats)
        pass


    def to_dict(self):
        """
        Save the molecule as a dictionary of arrays.
        """
        pass
    

    @staticmethod
    def from_dict(array_dict:Dict):
        """
        Create a Molecule from a dictionary of arrays.
        """
        pass
    

    def save(self, path:str):
        """
        Save the molecule to a npz file.
        """
        np.savez(path, **self.to_dict())

    @staticmethod
    def load(path:str):
        """
        Load the molecule from a npz file.
        """
        array_dict = np.load(path)
        return Molecule.from_dict(array_dict)

    @staticmethod
    def from_openmm_system(openmm_system, xyz, energy, gradient, partial_charges=None, energy_ref=None, gradient_ref=None, mapped_smiles=None, pdb=None):
        """
        Use an openmm system to obtain classical parameters and interaction tuples.
        If partial charges is None, the charges are obtained from the openmm system.
        If energy_ref and gradient_ref are None, they are also calculated from the openmm system.
        mapped_smiles and pdb have no effect on the system, are optional and only required for reproducibility.
        """
        pass