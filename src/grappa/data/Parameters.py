"""
Contains the output dataclass 'Parameters'.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from grappa.utils import tuple_indices

from .Molecule import Molecule

import pkgutil


@dataclass
class Parameters():
    """
    A parameter dict containing id tuples (corresponding to the atom_id passed in the atoms array) and np.ndarrays:
    
    {
    "atoms":np.array, the ids of the atoms in the molecule that correspond to the parameters. These are ids, not indices, i.e. they are not necessarily consecutive or start at zero.
    
    "{bond/angle}s":np.array of shape (#2/3-body-terms, 2/3), the ids of the atoms in the molecule that correspond to the parameters. The permutation symmetry of the n-body term is already divided out, i.e. this is the minimal set of parameters needed to describe the interaction.

    "{bond/angle}_k":np.array, the force constant of the interaction. In the same order as the id tuples in {bond/angle}s.

    "{bond/angle}_eq":np.array, the equilibrium distance of the interaction. In the same order as the id tuples in {bond/angle}s.

    
    "{proper/improper}s":np.array of shape (#4-body-terms, 4), the ids of the atoms in the molecule that correspond to the parameters. The central atom is at third position, i.e. index 2. For each entral atom, the array contains all cyclic permutation of the other atoms, i.e. 3 entries that all have different parameters in such a way that the total energy is invariant under cyclic permutation of the atoms.

    "{proper/improper}_ks":np.array of shape (#4-body-terms, n_periodicity), the fourier coefficients for the cos terms of torsion. These have the same order along axis 0 as the id tuples in {proper/improper}s. may be negative instead of the equilibrium dihedral angle (which is always set to zero). n_periodicity is a hyperparemter of the model and defaults to 6.

    "{proper/improper}_ns":np.array of shape (#4-body-terms, n_periodicity), the periodicities of the cos terms of torsion. These have the same order along axis 0 as the id tuples in {proper/improper}s. n_periodicity is a hyperparemter of the model and defaults to 6.

    "{proper/improper}_phases":np.array of shape (#4-body-terms, n_periodicity), the phases of the cos terms of torsion. These have the same order along axis 0 as the id tuples in {proper/improper}s. n_periodicity is a hyperparameter of the model and defaults to 6.

    }
    """
    atoms: np.ndarray

    bonds: np.ndarray
    bond_k: np.ndarray
    bond_eq: np.ndarray

    angles: np.ndarray
    angle_k: np.ndarray
    angle_eq: np.ndarray

    propers: np.ndarray
    proper_ks: np.ndarray
    proper_ns: np.ndarray
    proper_phases: np.ndarray

    impropers: Optional[np.ndarray] # optional because these are not needed for training grappa on classical parameters
    improper_ks: Optional[np.ndarray]
    improper_ns: Optional[np.ndarray]
    improper_phases: Optional[np.ndarray]

    @classmethod
    def from_dgl(cls, dgl_graph):
        """
        Assumes that the dgl graph has the following features:
            - 'ids' at node type n1 (these are the atom ids)
            - 'idxs' at node types n2, n3, n4, n4_improper (these are the indices of the atom ids the n1-'ids' vector and thus need to be converted to atom ids by ids = atoms[idxs])
        """
        pass

    @classmethod
    def from_openmm_system(cls, openmm_system, mol:Molecule, mol_is_sorted:bool=False):
        """
        Uses an openmm system to obtain classical parameters. The molecule is used to obtain the atom and interacion ids (not the openmm system!). The order of atom in the openmm system must be the same as in mol.atoms. Improper torsion parameters are not obtained from the openmm system.
        mol_is_sorted: if True, then it is assumed that the id tuples are sorted:
            bonds[i][0] < bonds[i][1] for all i
            angles[i][0] < angles[i][2] for all i
            propers[i][0] < propers[i][3] for all i
        """

        from openmm import HarmonicAngleForce, HarmonicBondForce, PeriodicTorsionForce
        from openmm import System

        # assert that the openmm system is a System:
        assert isinstance(openmm_system, System), "The openmm_system must be a openmm.System object."

        atoms = mol.atoms
        bonds = mol.bonds
        angles = mol.angles
        propers = mol.propers

        # initialize the arrays to zeros:
        bond_k = np.zeros(len(bonds), dtype=np.float32)
        bond_eq = np.zeros(len(bonds), dtype=np.float32)
        angle_k = np.zeros(len(angles), dtype=np.float32)
        angle_eq = np.zeros(len(angles), dtype=np.float32)
        proper_ks = np.zeros((len(propers), 6), dtype=np.float32)
        proper_ns = np.zeros((len(propers), 6), dtype=np.int32)
        proper_phases = np.zeros((len(propers), 6), dtype=np.float32)
        

        # NOTE: TRANSFORM THE UNITS AND THINK ABOUT TORSION PHASES! AND PERIODICITIES
        # iterate through bond, angle and proper torsion forces in openmm_system. then write the parameter to the corresponding position in the array.
        for force in openmm_system.getForces():
            if isinstance(force, HarmonicBondForce):
                for i in range(force.getNumBonds()):
                    bond_idx, atom1, atom2, bond_eq_, bond_k_ = force.getBondParameters(i)
                    bond = (atoms[atom1], atoms[atom2]) if atoms[atom1] < atoms[atom2] else (atoms[atom2], atoms[atom1])
                    bond_idx = bonds.index(bond)
                    bond_k[bond_idx] = bond_k_
                    bond_eq[bond_idx] = bond_eq_
        
            elif isinstance(force, HarmonicAngleForce):
                for i in range(force.getNumAngles()):
                    angle_idx, atom1, atom2, atom3, angle_eq_, angle_k_ = force.getAngleParameters(i)
                    angle = (atoms[atom1], atoms[atom2], atoms[atom3]) if atoms[atom1] < atoms[atom3] else (atoms[atom3], atoms[atom2], atoms[atom1])
                    angle_idx = angles.index(angle)
                    angle_k[angle_idx] = angle_k_
                    angle_eq[angle_idx] = angle_eq_

            # check whether the torsion is improper. if yes, skip it.
            elif isinstance(force, PeriodicTorsionForce):
                for i in range(force.getNumTorsions()):
                    torsion_idx, atom1, atom2, atom3, atom4, periodicity, phase, torsion_k = force.getTorsionParameters(i)
                    proper = (atoms[atom1], atoms[atom2], atoms[atom3], atoms[atom4]) if atoms[atom1] < atoms[atom4] else (atoms[atom4], atoms[atom3], atoms[atom2], atoms[atom1])
                    if proper in propers:
                        proper_idx = propers.index(proper)
                        proper_ks[proper_idx, periodicity-1] = torsion_k
                        proper_ns[proper_idx, periodicity-1] = periodicity
                        proper_phases[proper_idx, periodicity-1] = phase
            
        pass