#%%
"""
Contains the output dataclass 'Parameters'.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from grappa.utils import openmm_utils
from grappa import units
from grappa import constants
from grappa.constants import get_grappa_units_in_openmm
import torch
from dgl import DGLGraph
import warnings

from grappa.data.molecule import Molecule
import matplotlib.pyplot as plt
from grappa.utils.torch_utils import to_numpy

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

    "{proper/improper}_ks":np.array of shape (#4-body-terms, n_periodicity), the fourier coefficients for the cos terms of torsion. These have the same order along axis 0 as the id tuples in {proper/improper}s. The periodicity is given by 1 + the idx along axis=1, e.g. proper_ks[10,3] describes the term with n_per==4 of the torsion between the atoms propers[10]. May be negative instead of allowing an equilibrium dihedral angle (which is always set to zero). n_periodicity is a hyperparameter of the model and defaults to 6.

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
    def from_dgl(cls, g:DGLGraph, suffix:str='', check_eq_values:bool=True):
        """
        Assumes that the dgl graph has the following features:
            - 'ids' at node type n1 (these are the atom ids)
            - 'idxs' at node types n2, n3, n4, n4_improper (these are the indices of the atom ids the n1-'ids' vector and thus need to be converted to atom ids by ids = atoms[idxs])

        """

        # Extract the atom indices for each type of interaction
        # Assuming the indices are stored in edge data for 'n2', 'n3', and 'n4'
        # and that there's a mapping from indices to atom IDs available in node data for 'n1'
        atom_ids = g.nodes['n1'].data['ids'].detach().cpu().numpy()
        bonds = g.nodes['n2'].data['idxs'].detach().cpu().numpy()

        # Convert indices to atom IDs
        bonds = atom_ids[bonds]

        # Extract the classical parameters from the graph, assuming they have the suffix
        bond_k = g.nodes['n2'].data[f'k{suffix}'].detach().cpu().numpy()
        bond_eq = g.nodes['n2'].data[f'eq{suffix}'].detach().cpu().numpy()

        angle_k = g.nodes['n3'].data[f'k{suffix}'].detach().cpu().numpy()
        angle_eq = g.nodes['n3'].data[f'eq{suffix}'].detach().cpu().numpy()
        angles = g.nodes['n3'].data['idxs'].detach().cpu().numpy()
        angles = atom_ids[angles]

        # some checks:
        if check_eq_values:
            MAX_ANGLE = 45
            MAX_BOND_LENGTH = 0.5
            # if any angle is smaller than MAX_ANGLE degrees, notify the user
            if np.any(angle_eq < np.pi/180*MAX_ANGLE):
                n_smaller = np.sum(angle_eq < np.pi/180*MAX_ANGLE)
                raise RuntimeError(f"{n_smaller} angles are smaller than 20 degrees. This can lead to numerical instabilities in the model.\nThe smallest angle is {np.min(angle_eq)*180/np.pi} degrees at atom ids {angles[np.argmin(angle_eq)]}.\nnext: {angle_eq[np.argsort(angle_eq)[1:n_smaller+1]]*180/np.pi} degrees at atom ids {angles[np.argsort(angle_eq)[:n_smaller]]}.")
                    
            # if any bond eq length is smaller than MAX_BOND_LENGTH Angstrom, notify the user
            if np.any(bond_eq < MAX_BOND_LENGTH):
                n_smaller = np.sum(bond_eq < MAX_BOND_LENGTH)
                raise RuntimeError(f"{n_smaller} bond eq lengths are smaller than 0.5 Angstrom. This can lead to numerical instabilities in the model.\nThe smallest bond eq length is {np.min(bond_eq)} Angstrom at atom ids {bonds[np.argmin(bond_eq)]}.\nnext: {bond_eq[np.argsort(bond_eq)[1:n_smaller+1]]} Angstrom at atom ids {bonds[np.argsort(bond_eq)[:n_smaller]]}.")

        proper_ks = g.nodes['n4'].data[f'k{suffix}'].detach().cpu().numpy()
        # Assuming the phases are stored with a similar naming convention
        proper_phases = np.where(
            proper_ks >= 0.,
            np.zeros_like(proper_ks),
            np.zeros_like(proper_ks) + np.pi
        )
        proper_ks = np.abs(proper_ks)

        propers = g.nodes['n4'].data['idxs'].detach().cpu().numpy()
        propers = atom_ids[propers]


        improper_ks = g.nodes['n4_improper'].data[f'k{suffix}'].detach().cpu().numpy()
        improper_phases = np.where(
            improper_ks > 0,
            np.zeros_like(improper_ks),
            np.zeros_like(improper_ks) + np.pi
        )
        improper_ks = np.abs(improper_ks)

        impropers = atom_ids[g.nodes['n4_improper'].data['idxs'].detach().cpu().numpy()]

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
    def from_openmm_system(cls, openmm_system, mol:Molecule, mol_is_sorted:bool=False, allow_skip_improper:bool=False):
        """
        Uses an openmm system to obtain classical parameters. The molecule is used to obtain the atom and interacion ids (not the openmm system!). The order of atom in the openmm system must be the same as in mol.atoms. Improper torsion parameters are not obtained from the openmm system.
        mol_is_sorted: if True, then it is assumed that the id tuples are sorted:
            bonds[i][0] < bonds[i][1] for all i
            angles[i][0] < angles[i][2] for all i
            propers[i][0] < propers[i][3] for all i
            impropers: the central atom is inferred from connectivity, then it is put at place grappa.constants.IMPROPER_CENTRAL_IDX by invariant permutation.
        """

        # handle the units:
        grappa_units = get_grappa_units_in_openmm()
        BOND_K_UNIT = grappa_units['BOND_K']
        BOND_EQ_UNIT = grappa_units['BOND_EQ']
        ANGLE_K_UNIT = grappa_units['ANGLE_K']
        ANGLE_EQ_UNIT = grappa_units['ANGLE_EQ']
        TORSION_K_UNIT = grappa_units['TORSION_K']
        TORSION_PHASE_UNIT = grappa_units['TORSION_PHASE']

        from openmm import HarmonicAngleForce, HarmonicBondForce, PeriodicTorsionForce

        bonds = []
        bond_k = []
        bond_eq = []

        angles = []
        angle_k = []
        angle_eq = []

        torsions = []
        torsion_ks = []
        torsion_phases = []
        torsion_periodicities = []

        # iterate through bond, angle and proper torsion forces in openmm_system, convert to correct unit and append to list:

        for force in openmm_system.getForces():
            if isinstance(force, HarmonicBondForce):
                for i in range(force.getNumBonds()):
                    atom1, atom2, bond_eq_, bond_k_ = force.getBondParameters(i)
                   
                    # units:
                    bond_k_ = bond_k_.value_in_unit(BOND_K_UNIT)
                    bond_eq_ = bond_eq_.value_in_unit(BOND_EQ_UNIT)

                    # write to list:
                    bond_k.append(bond_k_)
                    bond_eq.append(bond_eq_)
                    bonds.append((atom1, atom2))
        

            elif isinstance(force, HarmonicAngleForce):
                for i in range(force.getNumAngles()):
                    atom1, atom2, atom3, angle_eq_, angle_k_ = force.getAngleParameters(i)

                    # units:
                    angle_k_ = angle_k_.value_in_unit(ANGLE_K_UNIT)
                    angle_eq_ = angle_eq_.value_in_unit(ANGLE_EQ_UNIT)

                    # write to list:
                    angle_k.append(angle_k_)
                    angle_eq.append(angle_eq_)
                    angles.append((atom1, atom2, atom3))


            # check whether the torsion is improper. if yes, skip it.
            elif isinstance(force, PeriodicTorsionForce):
                for i in range(force.getNumTorsions()):
                    atom1, atom2, atom3, atom4, periodicity, phase, torsion_k = force.getTorsionParameters(i)

                    # units:
                    torsion_k = torsion_k.value_in_unit(TORSION_K_UNIT)
                    phase = phase.value_in_unit(TORSION_PHASE_UNIT)
                    
                    # write to list:
                    torsion_ks.append(torsion_k)
                    torsion_phases.append(phase)
                    torsion_periodicities.append(periodicity)
                    torsions.append((atom1, atom2, atom3, atom4))


        return cls.from_lists(
            mol=mol,
            bonds=bonds,
            bond_k=bond_k,
            bond_eq=bond_eq,
            angles=angles,
            angle_k=angle_k,
            angle_eq=angle_eq,
            torsions=torsions,
            torsion_ks=torsion_ks,
            torsion_phases=torsion_phases,
            torsion_periodicities=torsion_periodicities,
            allow_skip_improper=allow_skip_improper,
            mol_is_sorted=mol_is_sorted,
        )
    


    @classmethod
    def from_lists(cls, mol, bonds, angles, torsions, bond_eq, angle_eq, bond_k, angle_k, torsion_ks, torsion_phases, torsion_periodicities, allow_skip_improper:bool=False, mol_is_sorted:bool=False):
        """
        Assume that the idxs in the bonds, angles, torsions lists correspond to entries at that idx position in mol.atoms.
        The lists must contain all bonds, angles and torsions in the molecule but may also contain more than that.
        Initializes the parameters from lists of interaction idxs and lists of parameters.
        For torsions, determines whether improper/proper and, if possible, expresses improper torsions with the central atom at position grappa.constants.IMPROPER_CENTRAL_IDX. If this is not possible, raises an error.
        """

        if not mol_is_sorted:
            # apply canonical ordering for bond and angle idxs:
            mol.sort()

        atoms = mol.atoms
        if not isinstance(atoms, np.ndarray):
            atoms = np.array(atoms).astype(np.int32)

        # convert to array:
        if not isinstance(bonds, np.ndarray):
            bonds = np.array(bonds).astype(np.int32)
        bonds = bonds.astype(np.int32)
        if not isinstance(angles, np.ndarray):
            angles = np.array(angles).astype(np.int32)
        bonds = bonds.astype(np.int32)

        if not isinstance(bond_eq, np.ndarray):
            bond_eq = np.array(bond_eq)
        if not isinstance(angle_eq, np.ndarray):
            angle_eq = np.array(angle_eq)
        if not isinstance(bond_k, np.ndarray):
            bond_k = np.array(bond_k)
        if not isinstance(angle_k, np.ndarray):
            angle_k = np.array(angle_k)


        assert len(bonds) == len(bond_eq) == len(bond_k), "The bond lists must have the same length."
        assert len(angles) == len(angle_eq) == len(angle_k), "The angle lists must have the same length."
        assert len(torsions) == len(torsion_ks), "The torsion lists must have the same length."

        # (these asserts can be removed if we allow bonds and angles with no energy contribution)
        assert len(bonds) >= len(mol.bonds), f"The bond lists must contain all bonds in the molecule but len(bonds)={len(bonds)} and len(mol.bonds)={len(mol.bonds)}."
        assert len(angles) >= len(mol.angles), f"The angle lists must contain all angles in the molecule but len(angles)={len(angles)} and len(mol.angles)={len(mol.angles)}."

        # assert that no bond, angle appears twice:
        assert len(np.unique(bonds, axis=0)) == len(bonds), f"The bond lists must not contain duplicates but {len(bonds) - len(np.unique(bonds, axis=0))} duplicates were found."
        assert len(np.unique(angles, axis=0)) == len(angles), f"The angle lists must not contain duplicates but {len(angles) - len(np.unique(angles, axis=0))} duplicates were found."

        # convert to atom ids:
        bonds = atoms[bonds]
        angles = atoms[angles]

        # apply canonical ordering for bond and angle idxs:
        bonds = np.sort(bonds, axis=1)
        angles = np.where((angles[:,0] < angles[:,2])[:, np.newaxis], angles, angles[:,::-1]) # reverse order where necessary

        # now find the indices of the molecule bonds and angles in the parameter lists:
        # better: create lookup dicts (scales O(1))
        bond_idxs = np.array([bonds.tolist().index(list(bond)) for bond in mol.bonds])
        angle_idxs = np.array([angles.tolist().index(list(angle)) for angle in mol.angles])

        # take those entries from the parameter lists:
        bond_eq = bond_eq[bond_idxs]
        angle_eq = angle_eq[angle_idxs]
        bond_k = bond_k[bond_idxs]
        angle_k = angle_k[angle_idxs]

        # now we are done with bonds and angles. For torsions, we need to differentiate between proper and improper torsions.

        # first initialize the arrays to zeros:
        proper_ks = np.zeros((len(mol.propers), constants.N_PERIODICITY_PROPER), dtype=np.float32)
        proper_phases = np.zeros((len(mol.propers), constants.N_PERIODICITY_PROPER), dtype=np.float32)
        improper_ks = np.zeros((len(mol.impropers), constants.N_PERIODICITY_IMPROPER), dtype=np.float32)
        improper_phases = np.zeros((len(mol.impropers), constants.N_PERIODICITY_IMPROPER), dtype=np.float32)

        # iterate through torsions and write the parameters to the corresponding position in the array.
        for torsion, torsion_k, phase, periodicity in zip(torsions, torsion_ks, torsion_phases, torsion_periodicities):
            if torsion_k == 0:
                continue

            # Enforce positive k here. (sign flip of k corresponds to phase shift by pi)
            phase = phase if torsion_k > 0 else (phase + np.pi) % (2*np.pi)
            torsion_k = torsion_k if torsion_k > 0 else -torsion_k

            # convert to mol indices
            torsion = tuple((atoms[torsion[i]] for i in range(4)))
            
            is_improper, central_atom_position = mol.is_improper(torsion)

            if not is_improper:
                if periodicity > constants.N_PERIODICITY_PROPER:
                    raise ValueError(f"The torsion {torsion} has a periodicity larger than {constants.N_PERIODICITY_PROPER}.")    
            
                # use that dihedral angle is invariant under reversal for canonical ordering:
                torsion = torsion if torsion[0] < torsion[3] else (torsion[3], torsion[2], torsion[1], torsion[0])
                try:
                    # better: create lookup dicts (scales O(1))
                    proper_idx = mol.propers.index(torsion)
                except ValueError:
                    raise ValueError(f"The torsion {torsion} is not included in the proper torsion list of the molecule.")
                
                if proper_ks[proper_idx, periodicity-1] != 0.:
                    # raise ValueError(f"The torsion {torsion} appears twice.")
                    if phase != proper_phases[proper_idx, periodicity-1]:
                        raise RuntimeError(f"The torsion {torsion} with n_periodicity={periodicity} appears twice with different phases: {phase} and {proper_phases[proper_idx, periodicity-1]}.")
                
                    # now we can simply add the ks since the energy is linear in k:
                    proper_ks[proper_idx, periodicity-1] += torsion_k

                else:
                    proper_ks[proper_idx, periodicity-1] = torsion_k # this is already made positive
                    proper_phases[proper_idx, periodicity-1] = phase

            ############################
            else:
                # the difficulty with impropers is that we have to store the improper id tuple such that the central atom is always at the same positions since grappa models are trained to use the information which atom is central upon prediction of the torsion parameters. (currently this is encoded by the position in the input)
                # we can do this by permuting the atoms such that the central atom is at position grappa and keeping track of the permutation.
                # the dihedral is invariant under order reversal and antisymmetric under permutation of the first and last or the second and third atom.
                # the symmetry leads to a symmetry of the energy term ( k cos(n phi + phase) ).
                # the antisymmetry of the dihedral, however, only leads to a symmetry of the energy term if the phase is either 0 or pi.

                if periodicity > constants.N_PERIODICITY_IMPROPER:
                    raise ValueError(f"The torsion {torsion} has a periodicity larger than {constants.N_PERIODICITY_IMPROPER}.")

                # the permutations above only mix 0 and 3 or 1 and 2, thus we can only permute the central atom from position 0 to 3 or from position 1 to 2:
                incompatible = False
                if central_atom_position in [0, 3] and constants.IMPROPER_CENTRAL_IDX not in [0,3]:
                    incompatible = True
                elif central_atom_position in [1, 2] and constants.IMPROPER_CENTRAL_IDX not in [1,2]:
                    incompatible = True
                if incompatible:
                    if allow_skip_improper:
                        continue
                    else:
                        raise RuntimeError(f"The central atom of the improper torsion {torsion} is at position {central_atom_position} in the molecule, but the constants.IMPROPER_CENTRAL_IDX is {constants.IMPROPER_CENTRAL_IDX}. Resolve this by using another value for constants.IMPROPER_CENTRAL_IDX or setting allow_skip_improper=True if you don't need the improper torsion parameters.")
                else:
                    # now we can find a permuted version of the improper torsion in the impropers list if the phase is either 0 or pi:
                    # use that the dihedral is invariant under order reversal and antisymmetric under permutation of the first and last or the second and third atom:
                    improper_found = False
                    for sign, permutation in [(1, [0,1,2,3]), (1, [3,2,1,0]), (-1, [0,2,1,3]), (-1, [3,1,2,0])]:
                        permuted_torsion = tuple([torsion[i] for i in permutation])

                        try:
                            improper_idx = mol.impropers.index(permuted_torsion)
                        except ValueError:
                            continue
                        if not np.isclose(phase, 0, atol=1e-2) and not np.isclose(phase, np.pi, atol=1e-2) and sign == -1:
                            # cannot allow antisymmetric permutation if phase is not 0 or pi (see above)
                            continue

                        k = sign * torsion_k

                        if improper_ks[improper_idx, periodicity-1] != 0.:
                            raise ValueError(f"The torsion {torsion} appears twice.")
                        
                        improper_ks[improper_idx, periodicity-1] = k
                        improper_phases[improper_idx, periodicity-1] = phase
                        improper_found = True
                        break

                    if not improper_found:
                        if allow_skip_improper:
                            continue
                        else:
                            raise RuntimeError(f"Allowed permutations of the improper torsion {torsion} with central atom position {central_atom_position} is not included in the improper torsion list of the molecule. The reason can be a phase that is neither 0 nor pi. phase/pi={phase/np.pi}. Resolve this by changing grappa.constants.IMPROPER_CENTRAL_IDX ({constants.IMPROPER_CENTRAL_IDX}) or setting allow_skip_improper=True if you don't need the improper torsion parameters.")
            ############################


        return cls(
            atoms=mol.atoms,
            bonds=mol.bonds,
            bond_k=bond_k,
            bond_eq=bond_eq,
            angles=mol.angles,
            angle_k=angle_k,
            angle_eq=angle_eq,
            propers=mol.propers,
            proper_ks=proper_ks,
            proper_phases=proper_phases,
            impropers=mol.impropers,
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
    

    def write_to_dgl(self, g:DGLGraph, n_periodicity_proper=constants.N_PERIODICITY_PROPER, n_periodicity_improper=constants.N_PERIODICITY_IMPROPER, suffix:str='_ref', allow_nan=True)->DGLGraph:
        """
        Write the parameters to a dgl graph.
        For torsion, we assume (and assert) that phases are only 0 or pi.
        """
        # write the classical parameters
        g.nodes['n2'].data[f'k{suffix}'] = torch.tensor(self.bond_k, dtype=torch.float32)
        g.nodes['n2'].data[f'eq{suffix}'] = torch.tensor(self.bond_eq, dtype=torch.float32)

        g.nodes['n3'].data[f'k{suffix}'] = torch.tensor(self.angle_k, dtype=torch.float32)
        g.nodes['n3'].data[f'eq{suffix}'] = torch.tensor(self.angle_eq, dtype=torch.float32)

        assert np.all((self.proper_ks >= 0) + np.isnan(self.proper_ks)), f"The proper torsion force constants must be positive but found the following values: {self.proper_ks[np.logical_not((self.proper_ks >= 0) + np.isnan(self.proper_ks))]}"

        if not np.all(np.isclose(self.proper_phases, 0, atol=1e-2) + np.isclose(self.proper_phases, np.pi, atol=1e-2) + np.isclose(self.proper_phases, 2*np.pi, atol=1e-2) + np.isnan(self.proper_phases)):
            if not allow_nan:
                raise ValueError(f"The proper torsion phases must be either 0 or pi or 2pi but found the following values: {self.proper_phases[np.logical_not(np.isclose(self.proper_phases, 0, atol=1e-2) + np.isclose(self.proper_phases, np.pi, atol=1e-2) + np.isclose(self.proper_phases, 2*np.pi, atol=1e-2) + np.isnan(self.proper_phases))]}")
            else:
                proper_ks = np.zeros_like(self.proper_ks) * np.nan

        else:
            proper_ks = np.where(
                np.isclose(self.proper_phases, 0, atol=1e-2) + np.isclose(self.proper_phases, 2*np.pi, atol=1e-2),
                self.proper_ks, -self.proper_ks)
        

        def correct_shape(x, shape1):
            """
            Helper for bringing the torsion parameters into the correct shape. Adds zeros or cuts off the end if necessary.
            """
            if x.shape[1] < shape1:
                # concat shape1 - x.shape[1] zeros to the right
                return torch.cat([x, torch.zeros_like(x[:,:(shape1 - x.shape[1])])], dim=1)
            elif x.shape[1] > shape1:
                w = Warning(f"n_periodicity ({shape1}) is smaller than the highest torsion periodicity found ({x.shape[1]}).")
                warnings.warn(w)
                return x[:,:shape1]
            else:
                return x
        
        g.nodes['n4'].data['k_ref'] = correct_shape(torch.tensor(proper_ks, dtype=torch.float32), n_periodicity_proper)

        assert np.all((self.improper_ks >= 0) + np.isnan(self.improper_ks)), f"The improper torsion force constants must be positive."
        if not np.all(np.isclose(self.improper_phases, 0, atol=1e-2) + np.isclose(self.improper_phases, np.pi, atol=1e-2) + np.isclose(self.improper_phases, 2*np.pi, atol=1e-2) + np.isnan(self.improper_phases)):
            if not allow_nan:
                raise ValueError("The improper torsion phases must be either 0 or pi or 2pi")
            else:
                improper_ks = np.zeros_like(self.improper_ks)
        else:
            improper_ks = Parameters.to_signed_k(self.improper_ks, self.improper_phases)

        g.nodes['n4_improper'].data['k_ref'] = correct_shape(torch.tensor(improper_ks, dtype=torch.float32), n_periodicity_improper)

        return g
    
    @staticmethod
    def to_signed_k(k:np.ndarray, phase:np.ndarray)->np.ndarray:
        """
        Converts the k and phase to a signed k.
        """
        k = to_numpy(k)
        phase = to_numpy(phase)
        assert np.all((k >= 0) + np.isnan(k)), f"The force constants must be positive."
        if not np.all(np.isclose(phase, 0, atol=1e-2) + np.isclose(phase, np.pi, atol=1e-2) + np.isclose(phase, 2*np.pi, atol=1e-2) + np.isnan(phase)):
            raise ValueError(f"The phases must be either 0 or pi or 2pi")
        
        return np.where(
                np.isclose(phase, 0, atol=1e-2) + np.isclose(phase, 2*np.pi, atol=1e-2),
                k, -k)

    @classmethod
    def get_nan_params(cls, mol:Molecule):
        """
        Returns a Parameters object with all parameters set to nan but in the correct shape.
        """
        atoms = np.array(mol.atoms).astype(np.int32)
        bonds = np.array(mol.bonds).astype(np.int32)
        angles = np.array(mol.angles).astype(np.int32)
        propers = np.array(mol.propers).astype(np.int32)
        impropers = np.array(mol.impropers).astype(np.int32)

        bond_k = np.full((len(bonds),), np.nan)
        bond_eq = np.full((len(bonds),), np.nan)

        angle_k = np.full((len(angles),), np.nan)
        angle_eq = np.full((len(angles),), np.nan)

        proper_ks = np.full((len(propers), constants.N_PERIODICITY_PROPER), np.nan)
        proper_phases = np.full((len(propers), constants.N_PERIODICITY_PROPER), np.nan)

        improper_ks = np.full((len(impropers), constants.N_PERIODICITY_IMPROPER), np.nan)
        improper_phases = np.full((len(impropers), constants.N_PERIODICITY_IMPROPER), np.nan)

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
    

    def __len__(self):
        return len(self.atoms)
    
    def plot(self, filename:str=None, **plot_args)->Union[None, Tuple]:
        """
        Save a plot of the parameters.
        """
        fig, ax = plot_parameters([self], **plot_args)

        if not filename is None:
            fig.savefig(filename)
            plt.close(fig)
            return None

        else:
            return fig, ax
        

    def compare_with(self, other, filename:str=None, **plot_args)->Union[None, Tuple]:
        """
        Compare the parameters with another Parameters object.
        """
        fig, ax = compare_parameters(self, other, **plot_args)

        if not filename is None:
            fig.savefig(filename)
            plt.close(fig)
            return None

        else:
            return fig, ax
        
    @classmethod
    def random(cls):
        """
        Returns random parameters for a toy molecule.
        """
        from grappa.data.molecule import Molecule
        mol = Molecule.random()
        params = cls.get_nan_params(mol)

        # for every param, pick randn like:

        params.bond_k = np.random.randn(len(params.bonds))*3 + 100
        params.bond_eq = np.random.randn(len(params.bonds)) + 10

        params.angle_k = np.random.randn(len(params.angles)) + 10
        params.angle_eq = (np.random.randn(len(params.angles))*0.5 + 3.14)

        params.proper_ks = np.abs(np.random.randn(len(params.propers), constants.N_PERIODICITY_PROPER))

        params.improper_ks = np.abs(np.random.randn(len(params.impropers), constants.N_PERIODICITY_IMPROPER))

        params.proper_phases = np.ones_like(params.proper_ks) * np.pi * 2
        params.improper_phases = np.ones_like(params.improper_ks) * np.pi * 2

        return params
    


################################################
# plotting utility


#%%
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from grappa.utils.torch_utils import to_numpy
from grappa import units
from matplotlib.colors import LogNorm


# set font to arial:
# plt.rc('font', family="Arial")



def compare_parameters(parameters_x: List[Parameters], parameters_y: List[Parameters], title: str=None, fontsize: int = 27, figsize: int = 7, s:float=4, xlabel:str=None, ylabel:str=None, scatter=True, log=True, gridsize:int=50, density:bool=False, exclude_idxs:List[np.ndarray]=None) -> Tuple[plt.Figure, plt.Axes]:
    """
    exclude_idxs: bonds and angles that involve these atoms will be ignored.
    """
    if not isinstance(parameters_x, list):
        parameters_x = [parameters_x]

    if not isinstance(parameters_y, list):
        parameters_y = [parameters_y]

    assert len(parameters_x) == len(parameters_y), "The number of parameter sets must be the same for x and y."

    # Extract parameters for x and y
    bond_eq_x = [to_numpy(params.bond_eq) for params in parameters_x]
    bond_k_x = [to_numpy(params.bond_k) for params in parameters_x]
    bond_eq_y = [to_numpy(params.bond_eq) for params in parameters_y]
    bond_k_y = [to_numpy(params.bond_k) for params in parameters_y]

    angle_eq_x = [to_numpy(params.angle_eq) for params in parameters_x]
    angle_k_x = [to_numpy(params.angle_k) for params in parameters_x]
    angle_eq_y = [to_numpy(params.angle_eq) for params in parameters_y]
    angle_k_y = [to_numpy(params.angle_k) for params in parameters_y]

    if exclude_idxs is not None:
        assert len(exclude_idxs) == len(parameters_x)
        bond_idxs = [to_numpy(params.bonds) for params in parameters_x]
        angle_idxs = [to_numpy(params.angles) for params in parameters_x]
        proper_idxs = [to_numpy(params.propers) for params in parameters_x]

        for i, idxs in enumerate(exclude_idxs):
            mask = ~np.isin(bond_idxs[i][:, :2], idxs).any(axis=1)
            bond_eq_x[i] = bond_eq_x[i][mask]
            bond_k_x[i] = bond_k_x[i][mask]
            bond_eq_y[i] = bond_eq_y[i][mask]
            bond_k_y[i] = bond_k_y[i][mask]

            mask = ~np.isin(angle_idxs[i][:, :3], idxs).any(axis=1)
            angle_eq_x[i] = angle_eq_x[i][mask]
            angle_k_x[i] = angle_k_x[i][mask]
            angle_eq_y[i] = angle_eq_y[i][mask]
            angle_k_y[i] = angle_k_y[i][mask]


    bond_eq_x = np.concatenate(bond_eq_x)
    bond_k_x = np.concatenate(bond_k_x)
    bond_eq_y = np.concatenate(bond_eq_y)
    bond_k_y = np.concatenate(bond_k_y)

    angle_eq_x = np.concatenate(angle_eq_x)
    angle_k_x = np.concatenate(angle_k_x)
    angle_eq_y = np.concatenate(angle_eq_y)
    angle_k_y = np.concatenate(angle_k_y)     
        

    n_periodicity = 3

    signed_proper_ks_x_n0 = np.concatenate([Parameters.to_signed_k(params.proper_ks, params.proper_phases)[:,0].flatten() for params in parameters_x], axis=0)
    signed_proper_ks_y_n0 = np.concatenate([Parameters.to_signed_k(params.proper_ks, params.proper_phases)[:,0].flatten() for params in parameters_y], axis=0)

    signed_proper_ks_x_n_rest = np.concatenate([Parameters.to_signed_k(params.proper_ks, params.proper_phases)[:,1:n_periodicity].flatten() for params in parameters_x], axis=0)
    signed_proper_ks_y_n_rest = np.concatenate([Parameters.to_signed_k(params.proper_ks, params.proper_phases)[:,1:n_periodicity].flatten() for params in parameters_y], axis=0)

    # Check shapes and raise ValueError if they do not match
    if bond_eq_x.shape != bond_eq_y.shape:
        raise ValueError(f"bond_eq_x.shape={bond_eq_x.shape}, bond_eq_y.shape={bond_eq_y.shape}")

    if bond_k_x.shape != bond_k_y.shape:
        raise ValueError(f"bond_k_x.shape={bond_k_x.shape}, bond_k_y.shape={bond_k_y.shape}")

    if angle_eq_x.shape != angle_eq_y.shape:
        raise ValueError(f"angle_eq_x.shape={angle_eq_x.shape}, angle_eq_y.shape={angle_eq_y.shape}")

    if angle_k_x.shape != angle_k_y.shape:
        raise ValueError(f"angle_k_x.shape={angle_k_x.shape}, angle_k_y.shape={angle_k_y.shape}")

    if signed_proper_ks_x_n0.shape != signed_proper_ks_y_n0.shape:
        raise ValueError(f"signed_proper_ks_x_n0.shape={signed_proper_ks_x_n0.shape}, signed_proper_ks_y_n0.shape={signed_proper_ks_y_n0.shape}")

    if signed_proper_ks_x_n_rest.shape != signed_proper_ks_y_n_rest.shape:
        raise ValueError(f"signed_proper_ks_x_n_rest.shape={signed_proper_ks_x_n_rest.shape}, signed_proper_ks_y_n_rest.shape={signed_proper_ks_y_n_rest.shape}")

    

    # convert rad to degree:
    angle_eq_x = units.Quantity(angle_eq_x, units.rad).value_in_unit(units.degree)
    angle_eq_y = units.Quantity(angle_eq_y, units.rad).value_in_unit(units.degree)

    angle_k_x = units.Quantity(angle_k_x, units.kcal_per_mol/units.rad/units.rad).value_in_unit(units.kcal_per_mol/units.degree/units.degree)
    angle_k_y = units.Quantity(angle_k_y, units.kcal_per_mol/units.rad/units.rad).value_in_unit(units.kcal_per_mol/units.degree/units.degree)

    # Create figure and axes
    fig, ax = plt.subplots(2, 3, figsize=(figsize * 3, figsize * 2))

    axes = ax.flatten()

    if density:
        import seaborn as sns
        try:
            sns.kdeplot(gridsize=gridsize, thresh=0.001, x=bond_eq_x, y=bond_eq_y, ax=axes[0], cmap="Blues", fill=True, alpha=1, norm=LogNorm() if log else None)
        except:
            pass
        try:
            sns.kdeplot(gridsize=gridsize, thresh=0.001, x=bond_k_x, y=bond_k_y, ax=axes[3], cmap="Blues", fill=True, alpha=1, norm=LogNorm() if log else None)
        except:
            pass
        try:
            sns.kdeplot(gridsize=gridsize, thresh=0.001, x=angle_eq_x, y=angle_eq_y, ax=axes[1], cmap="Blues", fill=True, alpha=1, norm=LogNorm() if log else None)
        except:
            pass
        try:
            sns.kdeplot(gridsize=gridsize, thresh=0.001, x=angle_k_x, y=angle_k_y, ax=axes[4], cmap="Blues", fill=True, alpha=1, norm=LogNorm() if log else None)
        except:
            pass
        try:
            sns.kdeplot(gridsize=gridsize, thresh=0.001, x=signed_proper_ks_x_n0, y=signed_proper_ks_y_n0, ax=axes[1], cmap="Blues", fill=True, alpha=1, norm=LogNorm() if log else None)
        except:
            pass
        try:
            sns.kdeplot(gridsize=gridsize, thresh=0.001, x=signed_proper_ks_x_n_rest, y=signed_proper_ks_y_n_rest, ax=axes[5], cmap="Blues", fill=True, alpha=1, norm=LogNorm() if log else None)
        except:
            pass

    if scatter:
        axes[0].scatter(bond_eq_x, bond_eq_y, color='blue', s=s)
        axes[1].scatter(angle_eq_x, angle_eq_y, color='blue', s=s)
        axes[2].scatter(signed_proper_ks_x_n0, signed_proper_ks_y_n0, color='blue', s=s)
        axes[3].scatter(bond_k_x, bond_k_y, color='blue', s=s)
        axes[4].scatter(angle_k_x, angle_k_y, color='blue', s=s)
        axes[5].scatter(signed_proper_ks_x_n_rest, signed_proper_ks_y_n_rest, color='blue', s=s) 

    # Titles and labels
    for i, ax in enumerate(axes):
        min_val = np.min([ax.get_xlim()[0], ax.get_ylim()[0]])
        max_val = np.max([ax.get_xlim()[1], ax.get_ylim()[1]])

        # ax.axline([min_val, min_val], [max_val, max_val], color='black', linestyle='--', linewidth=1)  # Add line

        ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', linewidth=1, alpha=1)  # Add line

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

        
        if not xlabel is None:
            ax.set_xlabel(xlabel, fontsize=fontsize)
        if not ylabel is None and i in [0,3]:
            ax.set_ylabel(ylabel, fontsize=fontsize)

        num_ticks = None

        if num_ticks:
            ax.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
            ax.yaxis.set_major_locator(plt.MaxNLocator(num_ticks))

        # make the ticks go into and out of the plot and make them larger:
        ax.tick_params(axis='both', which='major', direction='inout', length=10, width=1, labelsize=fontsize-3)
        
    # Units dictionary
    UNITS = {
        'bond_eq': r'$\AA$',
        'bond_k': r'kcal/mol/$\AA^2$',
        'angle_eq': 'deg',
        'angle_k': r'kcal/mol/deg$^2$',
        'proper_ks': r'kcal/mol',
        'improper_ks': r'kcal/mol',
    }

    # Set labels with units
    axes[0].set_title(f"Bond eq. [{UNITS['bond_eq']}]", fontsize=fontsize)
    axes[1].set_title(f"Angle eq. [{UNITS['angle_eq']}]", fontsize=fontsize)
    axes[2].set_title(f"Torsion k_0 [{UNITS['proper_ks']}]", fontsize=fontsize)
    axes[3].set_title(f"Bond k [{UNITS['bond_k']}]", fontsize=fontsize)
    axes[4].set_title(f"Angle k [{UNITS['angle_k']}]", fontsize=fontsize)
    axes[5].set_title(f"Torsion k_1-{n_periodicity} [{UNITS['proper_ks']}]", fontsize=fontsize)

    plt.tight_layout(pad=3.0)
    # plt.subplots_adjust(wspace=0.6, hspace=0.2)

    return fig, ax


def plot_parameters(parameters:List[Parameters], title=None, fontsize=27, figsize=7, compare_parameters:List[Parameters]=None, name="Grappa", compare_name="Reference"):

    if type(parameters) is not list:
        parameters = [parameters]

    if compare_parameters is not None:
        if type(compare_parameters) is not list:
            compare_parameters = [compare_parameters]

    # Create figure and axes with a 2-row, 3-column layout
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(figsize*2.8, figsize*2))
    
    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    n_periodicity = 3
    
    # Parameter data for plotting
    data = [
        np.concatenate([to_numpy(params.bond_eq) for params in parameters], axis=0),
        np.concatenate([to_numpy(params.angle_eq) for params in parameters], axis=0),
        np.concatenate([Parameters.to_signed_k(params.proper_ks, params.proper_phases)[:,0].flatten() for params in parameters], axis=0),
        np.concatenate([to_numpy(params.bond_k) for params in parameters], axis=0),
        np.concatenate([to_numpy(params.angle_k) for params in parameters], axis=0),
        np.concatenate([Parameters.to_signed_k(params.proper_ks, params.proper_phases)[:,1:n_periodicity].flatten() for params in parameters], axis=0),
    ]

    if compare_parameters is not None:
        data_compare = [
            np.concatenate([to_numpy(params.bond_eq) for params in compare_parameters], axis=0),
            np.concatenate([to_numpy(params.angle_eq) for params in compare_parameters], axis=0),
            np.concatenate([Parameters.to_signed_k(params.proper_ks, params.proper_phases)[:,0].flatten() for params in compare_parameters], axis=0),
            np.concatenate([to_numpy(params.bond_k) for params in compare_parameters], axis=0),
            np.concatenate([to_numpy(params.angle_k) for params in compare_parameters], axis=0),
            np.concatenate([Parameters.to_signed_k(params.proper_ks, params.proper_phases)[:,1:n_periodicity].flatten() for params in compare_parameters], axis=0),
        ]

        # convert rad to degree:
        data_compare[1] = units.Quantity(data_compare[1], units.rad).value_in_unit(units.degree)
        data_compare[4] = units.Quantity(data_compare[4], units.kcal_per_mol/units.rad/units.rad).value_in_unit(units.kcal_per_mol/units.degree/units.degree)


    # convert rad to degree:
    data[1] = units.Quantity(data[1], units.rad).value_in_unit(units.degree)

    data[4] = units.Quantity(data[4], units.kcal_per_mol/units.rad/units.rad).value_in_unit(units.kcal_per_mol/units.degree/units.degree)

    
    # Create violin plots for each parameter set
    for ax, param in zip(axes, data):
        if compare_parameters is not None:
            parts = ax.violinplot([param, data_compare[axes.tolist().index(ax)]], showmeans=False, showextrema=True, points=1000, bw_method=0.05)

            ax.set_xticks([1, 2])
            ax.set_xticklabels([name, compare_name], fontsize=fontsize-3)

        else:
            parts = ax.violinplot(param, showmeans=False, showextrema=True, points=1000, bw_method=0.05)

            # no x ticks:
            ax.set_xticks([])

        num_ticks = None

        if num_ticks:
            ax.yaxis.set_major_locator(plt.MaxNLocator(num_ticks))

        # make the ticks go into and out of the plot and make them larger:
        ax.tick_params(axis='both', which='major', direction='inout', length=10, width=1, labelsize=fontsize-3)

    # Units dictionary
    UNITS = {
        'bond_eq': r'$\AA$',
        'bond_k': r'kcal/mol/$\AA^2$',
        'angle_eq': 'deg',
        'angle_k': r'kcal/mol/deg$^2$',
        'proper_ks': r'kcal/mol',
        'improper_ks': r'kcal/mol',
    }

    # Set labels with units
    axes[0].set_title(f"Bond eq. [{UNITS['bond_eq']}]", fontsize=fontsize)
    axes[1].set_title(f"Angle eq. [{UNITS['angle_eq']}]", fontsize=fontsize)
    axes[2].set_title(f"Torsion k_0 [{UNITS['proper_ks']}]", fontsize=fontsize)
    axes[3].set_title(f"Bond k [{UNITS['bond_k']}]", fontsize=fontsize)
    axes[4].set_title(f"Angle k [{UNITS['angle_k']}]", fontsize=fontsize)
    axes[5].set_title(f"Torsion k_1-{n_periodicity} [{UNITS['proper_ks']}]", fontsize=fontsize)

    plt.tight_layout(pad=3.0)

    # Set the overall title and layout adjustment
    if not title is None:
        fig.suptitle(title, fontsize=fontsize)
        plt.subplots_adjust(top=0.85)  # Adjust the top padding to make room for title if necessary
    # plt.subplots_adjust(wspace=0.6, hspace=0.2)
    
    return fig, axes

# %%

if __name__ == '__main__':
    # test plotting
        
    # get some random parameters:
    random_parameters = [Parameters.random() for i in range(100)]

    # plot the parameters
    fig, ax = plot_parameters(random_parameters)

    fig.savefig('random_parameters.png')

    # close all:
    plt.close('all')
    # %%
    random_parameters2 = [Parameters.random() for i in range(100)]

    # compare the parameters
    fig, ax = compare_parameters(random_parameters, random_parameters2)

    fig.savefig('comparison_parameters.png')

    plt.close('all')



# %%
