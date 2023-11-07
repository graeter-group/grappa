"""
Contains the grappa input dataclass 'MolData', which is an extension of the dataclass 'Molecule' that contains conformational data and characterizations like smiles string or PDB file.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from grappa.data import Molecule, Parameters
from grappa.utils import openff_utils
import torch
from dgl import DGLGraph

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

    # classical forcafield energies:
    ff_energy: Optional[np.ndarray] = None
    ff_gradient: Optional[np.ndarray] = None


    def _validate(self):
        # parameter atoms must be same as molecule atoms
        # check shapes

        pass   
    
    def  __post_init__(self):

        self._validate()
    
    
    def to_dgl(self, max_element=53, exclude_feats:list[str]=[])->DGLGraph:
        """
        Converts the molecule to a dgl graph with node features. The elements are one-hot encoded.
        Also creates entries 'xyz', 'energy_ref' and 'gradient_ref' in the global node type g and in the atom node type n1, which contain the reference energy and gradient, respectively. The shapes are different than in the class attributes, namely (1, n_confs) and (n_atoms, n_confs, 3) respectively. (This is done because feature tensors must have len == num_nodes)

        The dgl graph has the following node types:
            - g: global
            - n1: atoms
            - n2: bonds
            - n3: angles
            - n4: propers
            - n4_improper: impropers
        The node type n1 carries the feature 'ids', which are the identifiers in self.atoms. The other interaction levels (n{>1}) carry the idxs (not ids) of the atoms as ordered in self.atoms as feature 'idxs'. These are not the identifiers but must be translated back to the identifiers using ids = self.atoms[idxs] after the forward pass.
        This also stores classical parameters except for improper torsions.
        """
        g = self.molecule.to_dgl(max_element=max_element, exclude_feats=exclude_feats)
        
        # write reference energy and gradient in the shape (1, n_confs) and (n_atoms, n_confs, 3) respectively
        g.nodes['g'].data['energy_ref'] = torch.tensor(self.reference_energy.reshape(1, -1), dtype=torch.float32)
        g.nodes['n1'].data['gradient_ref'] = torch.tensor(self.reference_gradient.transpose(1, 0, 2), dtype=torch.float32)

        # write positions in shape (n_atoms, n_confs, 3)
        g.nodes['n1'].data['xyz'] = torch.tensor(self.xyz.transpose(1, 0, 2), dtype=torch.float32)

        g = self.classical_parameters.write_to_dgl(g=g)

        return g
    

    def to_dict(self):
        """
        Save the molecule as a dictionary of arrays.
        """
        array_dict = dict()
        array_dict['xyz'] = self.xyz
        array_dict['energy'] = self.energy
        array_dict['gradient'] = self.gradient
        array_dict['reference_energy'] = self.reference_energy
        array_dict['reference_gradient'] = self.reference_gradient

        # optionals:
        if self.nonbonded_energy is not None:
            array_dict['nonbonded_energy'] = self.nonbonded_energy
        if self.nonbonded_gradient is not None:
            array_dict['nonbonded_gradient'] = self.nonbonded_gradient
        if self.ff_energy is not None:
            array_dict['ff_energy'] = self.ff_energy
        if self.ff_gradient is not None:
            array_dict['ff_gradient'] = self.ff_gradient

        moldict = self.molecule.to_dict()
        assert set(moldict.keys()).isdisjoint(array_dict.keys()), "Molecule and MolData have overlapping keys."
        array_dict.update(moldict)

        # remove bond, angle, proper, improper since these are stored in the molecule
        paramdict = {
            k: v for k, v in self.classical_parameters.to_dict() if k not in ['bond', 'angle', 'proper', 'improper']
            }

        assert set(paramdict.keys()).isdisjoint(array_dict.keys()), "Parameters and MolData have overlapping keys."

        array_dict.update(paramdict)

        return array_dict
    

    @classmethod
    def from_dict(cls, array_dict:Dict):
        """
        Create a Molecule from a dictionary of arrays.
        """
        pass
    

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

    @classmethod
    def from_openmm_system(cls, openmm_system, openmm_topology, xyz, energy, gradient, partial_charges=None, energy_ref=None, gradient_ref=None, mapped_smiles=None, pdb=None):
        """
        Use an openmm system to obtain classical parameters and interaction tuples.
        If partial charges is None, the charges are obtained from the openmm system.
        If energy_ref and gradient_ref are None, they are also calculated from the openmm system.
        mapped_smiles and pdb have no effect on the system, are optional and only required for reproducibility.
        """
        mol = Molecule.from_openmm_system(openmm_system=openmm_system, openmm_topology=openmm_topology, partial_charges=partial_charges)
        params = Parameters.from_openmm_system(openmm_system, mol=mol)

        self = cls(molecule=mol, classical_parameters=params, xyz=xyz, energy=energy, gradient=gradient, reference_energy=energy_ref, reference_gradient=gradient_ref, mapped_smiles=mapped_smiles, pdb=pdb)

        if self.reference_energy is None:
            # calculate reference energy and gradient from the openmm system using the partial charges provided
            pass

        return self
    

    @classmethod
    def from_smiles(cls, mapped_smiles, xyz, energy, gradient, partial_charges=None, energy_ref=None, gradient_ref=None, openff_forcefield='openff_unconstrained-1.2.0.offxml'):
        system, topology = openff_utils.get_openmm_system(mapped_smiles, openff_forcefield=openff_forcefield, partial_charges=partial_charges)
        return cls.from_openmm_system(openmm_system=system, openmm_topology=topology, xyz=xyz, energy=energy, gradient=gradient, partial_charges=partial_charges, energy_ref=energy_ref, gradient_ref=gradient_ref, mapped_smiles=mapped_smiles)