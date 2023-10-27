"""
Contains the grappa input dataclass 'Molecule' and output dataclass 'Parameters'.

XX further describe both
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


# TODO:compare to constants.py TopologyDict -> rename to Topology?
@dataclass
class Molecule():
    """Input dataclass for grappa parameter description

    Angles and proper dihedrals are optional and will be inferred from the bonds if None
    Additional features are a dict with name: array/list of shape n_atoms x feat_dim
    """
    atomnrs: list[int]
    elements: list[str]
    partial_charges: list[float]
    bonds: list[tuple[int,int]]
    impropers: list[tuple[int,int,int,int]]
    additional_features: dict[str,list]
    angles: Optional[list[tuple[int,int,int]]] = None
    propers: Optional[list[tuple[int,int,int,int]]] = None
    
    def _validate(self):
        # check the input for consistency
        pass   
    
    def  __post_init__(self):
        if self.angles is None:
            pass
            # build angles from bonds
        if self.propers is None:
            pass
            # build dihedrals from bonds

        self._validate()
    
# TODO: compare to constants.py ParamDict --> should this stay the same?
@dataclass
class Parameters():
    """
    A parameter dict containing index tuples (corresponding to the atom_idx passed in the atoms list) and np.ndarrays:
    
    {
    "atom_idxs":np.array, the indices of the atoms in the molecule that correspond to the parameters. In rising order and starts at zero.

    "atom_q":np.array, the partial charges of the atoms.

    "atom_sigma":np.array, the sigma parameters of the atoms.

    "atom_epsilon":np.array, the epsilon parameters of the atoms.

    
    "{bond/angle}_idxs":np.array of shape (#2/3-body-terms, 2/3), the indices of the atoms in the molecule that correspond to the parameters. The permutation symmetry of the n-body term is already divided out, i.e. this is the minimal set of parameters needed to describe the interaction.

    "{bond/angle}_k":np.array, the force constant of the interaction.

    "{bond/angle}_eq":np.array, the equilibrium distance of the interaction.   

    
    "{proper/improper}_idxs":np.array of shape (#4-body-terms, 4), the indices of the atoms in the molecule that correspond to the parameters. The central atom is at third position, i.e. index 2. For each entral atom, the array contains all cyclic permutation of the other atoms, i.e. 3 entries that all have different parameters in such a way that the total energy is invariant under cyclic permutation of the atoms.

    "{proper/improper}_ks":np.array of shape (#4-body-terms, n_periodicity), the fourier coefficients for the cos terms of torsion. may be negative instead of the equilibrium dihedral angle (which is always set to zero). n_periodicity is a hyperparemter of the model and defaults to 6.

    "{proper/improper}_ns":np.array of shape (#4-body-terms, n_periodicity), the periodicities of the cos terms of torsion. n_periodicity is a hyperparemter of the model and defaults to 6.

    "{proper/improper}_phases":np.array of shape (#4-body-terms, n_periodicity), the phases of the cos terms of torsion. n_periodicity is a hyperparameter of the model and defaults to 6.

    }
    """
    atom_idxs: np.ndarray
    atom_q: np.ndarray
    atom_sigma: np.ndarray
    atom_epsilon: np.ndarray
    bond_idxs: np.ndarray
    bond_k: np.ndarray
    bond_eq: np.ndarray
    angle_idxs: np.ndarray
    angle_k: np.ndarray
    angle_eq: np.ndarray
    proper_idxs: np.ndarray
    proper_ks: np.ndarray
    proper_ns: np.ndarray
    proper_phases: np.ndarray
    improper_idxs: np.ndarray
    improper_ks: np.ndarray
    improper_ns: np.ndarray
    improper_phases: np.ndarray
