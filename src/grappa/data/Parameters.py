"""
Contains the output dataclass 'Parameters'.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from grappa.utils import openmm_utils
from grappa import units as U
import torch
from dgl import DGLGraph

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

    "{proper/improper}_ks":np.array of shape (#4-body-terms, n_periodicity), the fourier coefficients for the cos terms of torsion. These have the same order along axis 0 as the id tuples in {proper/improper}s. The periodicity is given by 1 + the idx along axis=1, e.g. proper_ks[10,3] describes the term with n_per==4 of the torsion between the atoms propers[10]. May be negative instead of allowing an equilibrium dihedral angle (which is always set to zero). n_periodicity is a hyperparemter of the model and defaults to 6.

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
    proper_phases: np.ndarray

    impropers: Optional[np.ndarray] # optional because these are not needed for training grappa on classical parameters
    improper_ks: Optional[np.ndarray]
    improper_phases: Optional[np.ndarray]

    @classmethod
    def from_dgl(cls, g:DGLGraph, suffix:str=''):
        """
        Assumes that the dgl graph has the following features:
            - 'ids' at node type n1 (these are the atom ids)
            - 'idxs' at node types n2, n3, n4, n4_improper (these are the indices of the atom ids the n1-'ids' vector and thus need to be converted to atom ids by ids = atoms[idxs])
        """

        # Extract the atom indices for each type of interaction
        # Assuming the indices are stored in edge data for 'n2', 'n3', and 'n4'
        # and that there's a mapping from indices to atom IDs available in node data for 'n1'
        atom_ids = g.nodes['n1'].data['ids'].numpy()
        bonds = g.nodes['n2'].data['idxs'].numpy()

        # Convert indices to atom IDs
        bonds = atom_ids[bonds]

        # Extract the classical parameters from the graph, assuming they have the suffix
        bond_k = g.nodes['n2'].data[f'k{suffix}'].numpy()
        bond_eq = g.nodes['n2'].data[f'eq{suffix}'].numpy()

        if 'n3' in g.ntypes:
            angle_k = g.nodes['n3'].data[f'k{suffix}'].numpy()
            angle_eq = g.nodes['n3'].data[f'eq{suffix}'].numpy()
            angles = g.nodes['n3'].data['idxs'].numpy()
            angles = atom_ids[angles]
        else:
            angle_k = np.array([])
            angle_eq = np.array([])
            angles = np.array([])

        if 'n4' in g.ntypes:
            proper_ks = g.nodes['n4'].data[f'k{suffix}'].numpy()
            # Assuming the phases are stored with a similar naming convention
            proper_phases = np.where(
                proper_ks > 0,
                np.zeros_like(proper_ks),
                np.zeros_like(proper_ks) + np.pi
            )
            proper_ks = np.abs(proper_ks)
            propers = g.nodes['n4'].data['idxs'].numpy()
            propers = atom_ids[propers] 

        else:
            proper_ks = np.array([])
            proper_phases = np.array([])
            propers = np.array([])

        # Check if improper torsions are present
        if 'n4_improper' in g.ntypes:
            improper_ks = g.nodes['n4_improper'].data[f'k{suffix}'].numpy()
            improper_phases = np.where(
                improper_ks > 0,
                np.zeros_like(improper_ks),
                np.zeros_like(improper_ks) + np.pi
            )
            improper_ks = np.abs(improper_ks)
            impropers = atom_ids[g.nodes['n4_improper'].data['idxs'].numpy()]
        else:
            improper_ks = np.array([])
            improper_phases = np.array([])
            impropers = np.array([])

        return cls(
            atoms=atom_ids,
            bonds=bonds,
            bond_k=bond_k,
            bond_eq=bond_eq,
            angles=angles,
            angle_k=angle_k,
            angle_eq=angle_eq,
            propers=propers,
            proper_ks=proper_ks,
            proper_phases=proper_phases,
            impropers=impropers,
            improper_ks=improper_ks,
            improper_phases=improper_phases,
        )


    @classmethod
    def from_openmm_system(cls, openmm_system, mol:Molecule, mol_is_sorted:bool=False):
        """
        Uses an openmm system to obtain classical parameters. The molecule is used to obtain the atom and interacion ids (not the openmm system!). The order of atom in the openmm system must be the same as in mol.atoms. Improper torsion parameters are not obtained from the openmm system.
        mol_is_sorted: if True, then it is assumed that the id tuples are sorted:
            bonds[i][0] < bonds[i][1] for all i
            angles[i][0] < angles[i][2] for all i
            propers[i][0] < propers[i][3] for all i
            impropers: the central atom is inferred from connectivity, then it is put at place grappa.constants.IMPROPER_CENTRAL_IDX by invariant permutation.
        """
        from openmm import HarmonicAngleForce, HarmonicBondForce, PeriodicTorsionForce
        from openmm import System

        # assert that the openmm system is a System:
        assert isinstance(openmm_system, System), "The openmm_system must be a openmm.System object."

        if not mol_is_sorted:
            mol.sort()

        atoms = mol.atoms
        bonds = mol.bonds
        angles = mol.angles
        propers = mol.propers
        impropers = mol.impropers

        # initialize the arrays to zeros:
        bond_k = np.zeros(len(bonds), dtype=np.float32)
        bond_eq = np.zeros(len(bonds), dtype=np.float32)
        angle_k = np.zeros(len(angles), dtype=np.float32)
        angle_eq = np.zeros(len(angles), dtype=np.float32)
        proper_ks = np.zeros((len(propers), 6), dtype=np.float32)
        proper_phases = np.zeros((len(propers), 6), dtype=np.float32)
        improper_ks = np.zeros((len(impropers), 6), dtype=np.float32)
        improper_phases = np.zeros((len(impropers), 6), dtype=np.float32)


        # iterate through bond, angle and proper torsion forces in openmm_system. then write the parameter to the corresponding position in the array.
        for force in openmm_system.getForces():
            if isinstance(force, HarmonicBondForce):
                for i in range(force.getNumBonds()):
                    atom1, atom2, bond_eq_, bond_k_ = force.getBondParameters(i)
                    
                    bond = (atoms[atom1], atoms[atom2]) if atoms[atom1] < atoms[atom2] else (atoms[atom2], atoms[atom1])
                    bond_idx = bonds.index(bond)
                
                   
                    # units:
                    bond_k_ = bond_k_.value_in_unit(U.BOND_K_UNIT)
                    bond_eq_ = bond_eq_.value_in_unit(U.BOND_EQ_UNIT)

                    # write to array:
                    bond_k[bond_idx] = bond_k_
                    bond_eq[bond_idx] = bond_eq_
        

            elif isinstance(force, HarmonicAngleForce):
                for i in range(force.getNumAngles()):
                    atom1, atom2, atom3, angle_eq_, angle_k_ = force.getAngleParameters(i)

                    angle = (atoms[atom1], atoms[atom2], atoms[atom3]) if atoms[atom1] < atoms[atom3] else (atoms[atom3], atoms[atom2], atoms[atom1])
                
                    angle_idx = angles.index(angle)

                    # units:
                    angle_k_ = angle_k_.value_in_unit(U.ANGLE_K_UNIT)
                    angle_eq_ = angle_eq_.value_in_unit(U.ANGLE_EQ_UNIT)

                    # write to array:
                    angle_k[angle_idx] = angle_k_
                    angle_eq[angle_idx] = angle_eq_


            # check whether the torsion is improper. if yes, skip it.
            elif isinstance(force, PeriodicTorsionForce):
                for i in range(force.getNumTorsions()):
                    atom1, atom2, atom3, atom4, periodicity, phase, torsion_k = force.getTorsionParameters(i)

                    # units:
                    torsion_k = torsion_k.value_in_unit(U.TORSION_K_UNIT)
                    phase = phase.value_in_unit(U.TORSION_PHASE_UNIT)
                    
                    # write to array: Enforce positive k here.
                    phase = phase if torsion_k > 0 else (phase + np.pi) % (2*np.pi)
                    torsion_k = torsion_k if torsion_k > 0 else -torsion_k

                    torsion = (atoms[atom1], atoms[atom2], atoms[atom3], atoms[atom4])
                    
                    is_improper, _ = mol.is_improper(torsion)
                    if not is_improper:
                        torsion = torsion if atoms[atom1] < atoms[atom4] else (atoms[atom4], atoms[atom3], atoms[atom2], atoms[atom1])
                        try:
                            proper_idx = propers.index(torsion)
                        except ValueError:
                            raise ValueError(f"The torsion {torsion} is not included in the proper torsion list of the molecule.")
                        
                        proper_ks[proper_idx, periodicity-1] = torsion_k if torsion_k > 0 else -torsion_k
                        proper_phases[proper_idx, periodicity-1] = phase

                        # set k to zero in the force:
                        force.setTorsionParameters(i, atom1, atom2, atom3, atom4, periodicity, phase, 0)

                    else:
                        # since we cannot translate from position 0 to position 2 (we can only transport the central atom position from 0 to 3 or 1 to 2), simply assign zero. We can learn on improper contributions instead.

                        continue



        return cls(
            atoms=atoms,
            bonds=bonds,
            bond_k=bond_k,
            bond_eq=bond_eq,
            angles=angles,
            angle_k=angle_k,
            angle_eq=angle_eq,
            propers=propers,
            proper_ks=proper_ks,
            proper_phases=proper_phases,
            impropers=impropers,
            improper_ks=improper_ks,
            improper_phases=improper_phases,
        )
    
    def to_dict(self):
        """
        Save the parameters as a dictionary of arrays.
        """
        d = {
            'atoms': self.atoms,
            'bonds': self.bonds,
            'bond_k': self.bond_k,
            'bond_eq': self.bond_eq,
            'angles': self.angles,
            'angle_k': self.angle_k,
            'angle_eq': self.angle_eq,
            'propers': self.propers,
            'proper_ks': self.proper_ks,
            'proper_phases': self.proper_phases,
        }
        if self.impropers is not None:
            d['impropers'] = self.impropers
            d['improper_ks'] = self.improper_ks
            d['improper_phases'] = self.improper_phases

        return d


    @classmethod
    def from_dict(cls, array_dict:Dict):
        """
        Create a Parameters object from a dictionary of arrays.
        """
        return cls(**array_dict)
    

    def write_to_dgl(self, g:DGLGraph)->DGLGraph:
        """
        Write the parameters to a dgl graph.
        For torsion, assume that phases are only 0 or pi.
        """
        # write the classical parameters
        g.nodes['n2'].data['k_ref'] = torch.tensor(self.bond_k, dtype=torch.float32)
        g.nodes['n2'].data['eq_ref'] = torch.tensor(self.bond_eq, dtype=torch.float32)

        if 'n3' in g.ntypes:
            g.nodes['n3'].data['k_ref'] = torch.tensor(self.angle_k, dtype=torch.float32)
            g.nodes['n3'].data['eq_ref'] = torch.tensor(self.angle_eq, dtype=torch.float32)

        if 'n4' in g.ntypes:
            proper_ks = np.where(
                np.isclose(self.proper_phases, 0, atol=1e-2) + np.isclose(self.proper_phases, 2*np.pi, atol=1e-2),
                self.proper_ks, -self.proper_ks)
            g.nodes['n4'].data['k_ref'] = torch.tensor(proper_ks, dtype=torch.float32)

        if 'n4_improper' in g.ntypes:
            improper_ks = np.where(
                np.isclose(self.improper_phases, 0, atol=1e-2) + np.isclose(self.improper_phases, 2*np.pi, atol=1e-2),
                self.improper_ks, -self.improper_ks)
            g.nodes['n4_improper'].data['k_ref'] = torch.tensor(improper_ks, dtype=torch.float32)

        return g