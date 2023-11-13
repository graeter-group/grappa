"""
Contains the grappa input dataclass 'Molecule' and output dataclass 'Parameters'.

XX further describe both
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Optional
import numpy as np
from pathlib import Path
import json

from grappa.run.run_utils import dataclass_from_dict

# TODO:compare to constants.py TopologyDict -> rename to Topology?
@dataclass
class Molecule():
    """Input dataclass for grappa parameter description

    Angles and proper dihedrals are optional and will be inferred from the bonds if None
    Additional features are a dict with name: array/list of shape n_atoms x feat_dim
    """
    atoms: list[int]
    bonds: list[tuple[int,int]]
    impropers: list[tuple[int,int,int,int]]
    atomic_nrs: list[int]
    partial_charges: list[float]
    epsilons: list[float]
    sigmas: list[float]
    additional_features: dict[str,list] = field(default_factory=dict)
    angles: Optional[list[tuple[int,int,int]]] = None
    propers: Optional[list[tuple[int,int,int,int]]] = None
    
    def _validate(self):
        # check the input for consistency
        # TODO: compare to current input validation
        pass   

    
    def  __post_init__(self):
        #TODO: check whether this does what is is supposed to do
        if self.angles is None or self.propers is None:
            # generate bound_to dict
            bound_to = {}
            for bond in self.bonds:
                for i,atomnr in enumerate(bond):
                    if bound_to.get(atomnr):
                        bound_to[atomnr].append(bond[1-i])
                        bound_to[atomnr].sort(key=int)
                    else:
                        bound_to[atomnr] = [bond[1-i]]

            if self.angles is None:
                self.angles = []
                # build angles from bound_to
                for atomnr, neighbors in bound_to.items():
                    for i, n1 in enumerate(neighbors):
                        for ii, n2 in enumerate(neighbors[:i]):
                            self.angles.append([n1,atomnr,n2])

            if self.propers is None:
                self.propers = []
                # build proper dihedrals from bound_to
                for atomnr, neighbors in bound_to.items():
                    for i, center in enumerate(neighbors):
                        for ii, outer1 in enumerate(neighbors[:i]):
                            for iii,outer2 in enumerate(bound_to.get(center)):
                                if atomnr != outer2:
                                    self.propers.append([outer1,atomnr,center,outer2])
            

        self._validate()

    def to_json(self, path: Path) -> None:
        """Write dict to file according to JSON format."""
        with open(path,'w') as f:
            json.dump(asdict(self),f)

    @classmethod
    def from_json(cls, path: Path) -> Molecule:
        """Return JSON file content as dict."""
        with open(path, "r") as f:
            data = json.load(f)      
        mol = dataclass_from_dict(Molecule, data)     
        return mol
    
    @classmethod
    def create_empty(cls) -> Molecule:
        "Return Molecule with empty attributes."
        return Molecule([],[],[],[],[],[],[],{})


    
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
