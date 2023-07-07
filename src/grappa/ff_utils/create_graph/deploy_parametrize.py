
# NOTE: If we also have e.g. water, we only want to parametrize some part of the topology. how can we do this in openmm?

# supress openff warning:
import logging
logging.getLogger("openff").setLevel(logging.ERROR)
from openff.toolkit.topology import Molecule

import openmm
import numpy as np
import tempfile
import os.path
import torch
import dgl
import openff.toolkit.topology
from typing import Union
from pathlib import Path
from typing import List, Tuple, Dict, Union, Callable

from . import utils, find_radical
from .. import units, charge_models
from ..classical_ff.parametrize import add_radical_residues

import copy

RESIDUES = units.RESIDUES

def write_is_radical(g:dgl.DGLGraph, is_radical:Union[List, np.ndarray, torch.tensor])->dgl.DGLGraph:

    if not g.num_nodes["n1"] == len(is_radical):
        raise ValueError(f"Number of nodes in g: {g.num_nodes['n1']} does not match number of is_radical: {len(is_radical)}")
    
    is_radical = np.array(is_radical)
    if not np.all(np.isin(is_radical, [0,1])):
        raise ValueError(f"Is_radical must be a list of 0s and 1s, but got {is_radical}")
    
    is_radical = torch.tensor(is_radical, dtype=torch.float32)
    if len(is_radical.shape) == 1:
        is_radical = is_radical.unsqueeze(dim=1)
    assert len(is_radical.shape) == 2 and is_radical.shape[1] == 1
    g.nodes["n1"].data["is_radical"] = is_radical

    return g

def write_partial_charges(g:dgl.DGLGraph, partial_charges:Union[List, np.ndarray, torch.tensor])->dgl.DGLGraph:
    if not g.num_nodes["n1"] == len(partial_charges):
        raise ValueError(f"Number of nodes in g: {g.num_nodes['n1']} does not match number of partial charges: {len(partial_charges)}")
    g.nodes["n1"].data["q"] = torch.tensor(partial_charges)
    g.nodes["n1"].data["q_ref"] = torch.tensor(partial_charges) # this is used as input feature for the model, the other one as output parameter. this is only for backwards compatibility

    return g



def write_lj(g:dgl.DGLGraph, sigmas:Union[List, np.ndarray, torch.tensor], epsilons:Union[List, np.ndarray, torch.tensor])->dgl.DGLGraph:
    """
    The sigma and epsilon list must be in the same order as the nodes in g.
    Writes the sigma and epsilon values to the graph.
    (The model does not need this information)
    """

    if not g.num_nodes["n1"] == len(sigmas):
        raise ValueError(f"Number of nodes in g: {g.num_nodes['n1']} does not match number of sigmas: {len(sigmas)}")
    if not g.num_nodes["n1"] == len(epsilons):
        raise ValueError(f"Number of nodes in g: {g.num_nodes['n1']} does not match number of epsilons: {len(epsilons)}")
    
    g.nodes["n1"].data["sigma"] = torch.tensor(sigmas)
    g.nodes["n1"].data["epsilon"] = torch.tensor(epsilons)

    return g

def write_parameters(g, topology, forcefield, get_charges, allow_radicals:bool=True):
    """
    openmm_top: openmm topology
    forcefield: openmm forcefield

    writes partial_charges, sigma and epsilon in the graph. if allow_radical is True, also writes is_radical in the graph. This is done for compatibility with other input/output types: We wish to store all information in the dgl graph at some point.
    """

    if allow_radicals:
        forcefield = copy.deepcopy(forcefield)
        forcefield = add_radical_residues(forcefield, topology)


    # will get modified in-place by get_charges, needed for setting is_radical
    radical_indices = []

    manual_charges = None
    if not get_charges is None:
        manual_charges = get_charges(topology, radical_indices=radical_indices)
        manual_charges = torch.tensor(manual_charges, dtype=torch.float32)


    sys = forcefield.createSystem(topology=topology)

    # PARAMETRISATION START
    atom_lookup = {
        idxs.detach().numpy().item(): position
        for position, idxs in enumerate(g.nodes["n1"].data["idxs"])
    }

    g.nodes["n1"].data["residue"] = torch.zeros(
        len(atom_lookup.keys()), len(RESIDUES), dtype=torch.float32,
    )
            
    g.nodes["n1"].data["is_radical"] = torch.zeros(
        len(atom_lookup.keys()), 1, dtype=torch.float32,
    )

    # get parameters from the forcefield in openmm representation:
    for force in sys.getForces():
        name = force.__class__.__name__
        if "NonbondedForce" in name:
            assert (
                force.getNumParticles()
                == g.number_of_nodes("n1")
            )

            g.nodes["n1"].data["epsilon"] = torch.zeros(
                force.getNumParticles(), 1, dtype=torch.float32,
            )

            g.nodes["n1"].data["sigma"] = torch.zeros(
                force.getNumParticles(), 1, dtype=torch.float32,
            )

            g.nodes["n1"].data["q"] = torch.zeros(
                force.getNumParticles(), 1, dtype=torch.float32,
            )

            for idx in range(force.getNumParticles()):

                charge, sigma, epsilon = force.getParticleParameters(idx)

                position = atom_lookup[idx]

                g.nodes["n1"].data["epsilon"][position] = epsilon.value_in_unit(
                    units.ENERGY_UNIT,
                )
                g.nodes["n1"].data["sigma"][position] = sigma.value_in_unit(
                    units.DISTANCE_UNIT,
                )

                if not manual_charges is None:
                    g.nodes["n1"].data["q"][position] = manual_charges[position]
                else:
                    g.nodes["n1"].data["q"][position] = charge.value_in_unit(
                        units.CHARGE_UNIT,
                    )
    
    g.nodes["n1"].data["q_ref"] = g.nodes["n1"].data["q"] # this is used as input feature for the model, the other one as output parameter. this is only for backwards compatibility

    return g


# difficult case: only bond lists are given, no valid pdb, maybe even only a substructure, and no partial charges / lj parameters areprovided. (then, construct openmm topology by hand and delete and cap end residues where invalidities occur. keep track of the indices somehow.)
# residue index needed?
# first write the pipeline for input given as pdb



def openmm_system_from_params(param_dict:dict, classical_ff, topology, allow_radicals:bool=True, system_kwargs=None)->openmm.System:
    """
    Returns an openmm system initialized from a parameter dict.
    Indices in the system will correspond to thge order of masses. Dictionary entries must be of the form:

    {
        "atom_idxs":np.array, the indices of the atoms in the molecule that correspond to the parameters. In rising order and starts at zero.

        "atom_q":np.array, the partial charges of the atoms.

        "atom_sigma":np.array, the sigma parameters of the atoms.

        "atom_epsilon":np.array, the epsilon parameters of the atoms.

        optional (if 'm' or 'mass' in the graph data keys, m has precedence over mass):
            "atom_mass":np.array, the masses of the atoms in atomic units.

        
        "{bond/angle}_idxs":np.array of shape (#2/3-body-terms, 2/3), the indices of the atoms in the molecule that correspond to the parameters. The permutation symmetry of the n-body term is already divided out, i.e. this is the minimal set of parameters needed to describe the interaction.

        "{bond/angle}_k":np.array, the force constant of the interaction.

        "{bond/angle}_eq":np.array, the equilibrium distance of the interaction.   

        
        "{proper/improper}_idxs":np.array of shape (#4-body-terms, 4), the indices of the atoms in the molecule that correspond to the parameters. The permutation symmetry of the n-body term is already divided out, i.e. this is the minimal set of parameters needed to describe the interaction.
        "{proper/improper}_ks":np.array of shape (#4-body-terms, n_periodicity), the fourier coefficients for the cos terms of torsion. may be negative instead of the equilibrium dihedral angle (which is always set to zero). n_periodicity is a hyperparemter of the model and defaults to 6.
    }
    """



    # =============================================================================
    # UNITS
    # =============================================================================
    from openmm import unit
    from openmm.unit import Quantity

    OPENMM_LENGTH_UNIT = unit.nanometer
    OPENMM_ANGLE_UNIT = unit.radian
    OPENMM_ENERGY_UNIT = unit.kilojoule_per_mole
    OPENMM_MASS_UNIT = unit.dalton

    OPENMM_BOND_EQ_UNIT = OPENMM_LENGTH_UNIT
    OPENMM_ANGLE_EQ_UNIT = OPENMM_ANGLE_UNIT
    OPENMM_TORSION_K_UNIT = OPENMM_ENERGY_UNIT
    OPENMM_TORSION_PHASE_UNIT = OPENMM_ANGLE_UNIT
    OPENMM_BOND_K_UNIT = OPENMM_ENERGY_UNIT / (OPENMM_LENGTH_UNIT**2)
    OPENMM_ANGLE_K_UNIT = OPENMM_ENERGY_UNIT / (OPENMM_ANGLE_UNIT**2)
    OPENMM_SIGMA_UNIT = OPENMM_LENGTH_UNIT
    OPENMM_EPSILON_UNIT = OPENMM_ENERGY_UNIT
    OPENMM_CHARGE_UNIT = unit.elementary_charge

    from .. import units as grappa_units

    # =============================================================================
    # SYSTEM
    # =============================================================================
    
    if allow_radicals:
        # copy the force field as to not modify it if residues are added
        classical_ff = copy.deepcopy(classical_ff)
        # add radical residues if necessary
        classical_ff = add_radical_residues(classical_ff, topology)


    sys_kwargs = {}
    if not system_kwargs is None:
        sys_kwargs.update(system_kwargs)

    sys = classical_ff.createSystem(topology=topology, **sys_kwargs)
    
    # NONBONDED
    # this must be treated differently: we have to use one-four interaction correctly, so just re-use the nonbonded force from the forcefield.
    charges = Quantity(param_dict["atom_q"], grappa_units.CHARGE_UNIT).value_in_unit(OPENMM_CHARGE_UNIT)
    sigmas = Quantity(param_dict["atom_sigma"], grappa_units.DISTANCE_UNIT).value_in_unit(OPENMM_SIGMA_UNIT)
    epsilons = Quantity(param_dict["atom_epsilon"], grappa_units.ENERGY_UNIT).value_in_unit(OPENMM_EPSILON_UNIT)

    # now remove all but the nonbonded forces:
    i = 0
    while i < sys.getNumForces():
        force = sys.getForce(i)
        if not "Nonbonded" in force.__class__.__name__:
            sys.removeForce(i)
        else:
            # replace nonbonded parameters:
            for j in range(force.getNumParticles()):
                charge, sigma, epsilon = charges[j], sigmas[j], epsilons[j]
                force.setParticleParameters(index=j, charge=charge, sigma=sigma, epsilon=epsilon)
            i += 1

    # BONDS
    bonds = param_dict["bond_idxs"]
    bond_ks = Quantity(param_dict["bond_k"], grappa_units.FORCE_CONSTANT_UNIT).value_in_unit(OPENMM_BOND_K_UNIT)
    bond_eqs = Quantity(param_dict["bond_eq"], grappa_units.DISTANCE_UNIT).value_in_unit(OPENMM_BOND_EQ_UNIT)

    bond_force = openmm.HarmonicBondForce()
    for idxs, k, eq in zip(bonds, bond_ks, bond_eqs):
        idx1, idx2 = idxs
        bond_force.addBond(idx1, idx2, eq, k)
    sys.addForce(bond_force)

    # ANGLES
    angles = param_dict["angle_idxs"]
    angle_ks = Quantity(param_dict["angle_k"], grappa_units.ANGLE_FORCE_CONSTANT_UNIT).value_in_unit(OPENMM_ANGLE_K_UNIT)
    angle_eqs = Quantity(param_dict["angle_eq"], grappa_units.ANGLE_UNIT).value_in_unit(OPENMM_ANGLE_EQ_UNIT)

    angle_force = openmm.HarmonicAngleForce()
    for idxs, k, eq in zip(angles, angle_ks, angle_eqs):
        idx1, idx2, idx3 = idxs
        angle_force.addAngle(idx1, idx2, idx3, eq, k)
    sys.addForce(angle_force)

    # TORSIONS
    torsion_force = openmm.PeriodicTorsionForce()
    for torsion_type in ["proper", "improper"]:

        torsions = param_dict[f"{torsion_type}_idxs"]
        torsion_ks = Quantity(param_dict[f"{torsion_type}_ks"], grappa_units.ENERGY_UNIT).value_in_unit(OPENMM_TORSION_K_UNIT)
        torsion_phases = Quantity(param_dict[f"{torsion_type}_phases"], grappa_units.ANGLE_UNIT).value_in_unit(OPENMM_TORSION_PHASE_UNIT)
        torsion_periodicities = param_dict[f"{torsion_type}_ns"]
        
        for idxs, ks, ns, phases in zip(torsions, torsion_ks, torsion_periodicities, torsion_phases):
            idx1, idx2, idx3, idx4 = idxs

            for i, k in enumerate(ks):
                if k == 0.0:
                    continue

                phase = phases[i]
                n = ns[i]

                torsion_force.addTorsion(idx1, idx2, idx3, idx4, n, phase, k)
        
    sys.addForce(torsion_force)

    return sys



def write_in_openmm_system(param_dict:dict, classical_ff, topology, allow_radicals:bool=True, system_kwargs=None)->openmm.System:
    """
    The dictionary must not be in symmetry-reduced form! I.e. contain all terms, even if they are redundant.
    Returns an openmm system initialized by a classical force field with parameters overwritten by the param_dict. 
    The param_dict only overwrites parameters with the indices given in the param_dict.

    Dictionary entries must be of the form:

    {
        "atom_idxs":np.array, the indices of the atoms in the molecule that correspond to the parameters. In rising order and starts at zero.

        "atom_q":np.array, the partial charges of the atoms.

        "atom_sigma":np.array, the sigma parameters of the atoms.

        "atom_epsilon":np.array, the epsilon parameters of the atoms.

        optional (if 'm' or 'mass' in the graph data keys, m has precedence over mass):
            "atom_mass":np.array, the masses of the atoms in atomic units.

        
        "{bond/angle}_idxs":np.array of shape (#2/3-body-terms, 2/3), the indices of the atoms in the molecule that correspond to the parameters. The permutation symmetry of the n-body term is already divided out, i.e. this is the minimal set of parameters needed to describe the interaction.

        "{bond/angle}_k":np.array, the force constant of the interaction.

        "{bond/angle}_eq":np.array, the equilibrium distance of the interaction.   

        
        "{proper/improper}_idxs":np.array of shape (#4-body-terms, 4), the indices of the atoms in the molecule that correspond to the parameters. The permutation symmetry of the n-body term is already divided out, i.e. this is the minimal set of parameters needed to describe the interaction.
        "{proper/improper}_ks":np.array of shape (#4-body-terms, n_periodicity), the fourier coefficients for the cos terms of torsion. may be negative instead of the equilibrium dihedral angle (which is always set to zero). n_periodicity is a hyperparemter of the model and defaults to 6.
    }
    """

    raise NotImplementedError("This function is not yet implemented correctly.")

    # =============================================================================
    # UNITS
    # =============================================================================
    from openmm import unit
    from openmm.unit import Quantity

    OPENMM_LENGTH_UNIT = unit.nanometer
    OPENMM_ANGLE_UNIT = unit.radian
    OPENMM_ENERGY_UNIT = unit.kilojoule_per_mole
    OPENMM_MASS_UNIT = unit.dalton

    OPENMM_BOND_EQ_UNIT = OPENMM_LENGTH_UNIT
    OPENMM_ANGLE_EQ_UNIT = OPENMM_ANGLE_UNIT
    OPENMM_TORSION_K_UNIT = OPENMM_ENERGY_UNIT
    OPENMM_TORSION_PHASE_UNIT = OPENMM_ANGLE_UNIT
    OPENMM_BOND_K_UNIT = OPENMM_ENERGY_UNIT / (OPENMM_LENGTH_UNIT**2)
    OPENMM_ANGLE_K_UNIT = OPENMM_ENERGY_UNIT / (OPENMM_ANGLE_UNIT**2)
    OPENMM_SIGMA_UNIT = OPENMM_LENGTH_UNIT
    OPENMM_EPSILON_UNIT = OPENMM_ENERGY_UNIT
    OPENMM_CHARGE_UNIT = unit.elementary_charge

    from .. import units as grappa_units

    # =============================================================================
    # SYSTEM
    # =============================================================================
    
    if allow_radicals:
        # copy the force field as to not modify it if residues are added
        classical_ff = copy.deepcopy(classical_ff)
        # add radical residues if necessary
        classical_ff = add_radical_residues(classical_ff, topology)


    sys_kwargs = {}
    if not system_kwargs is None:
        sys_kwargs.update(system_kwargs)

    sys = classical_ff.createSystem(topology=topology, **sys_kwargs)

    # NOTE:
    # indices off the forces do not necessarily correspond to the topology indices.
    # try this out for water system

    for force in sys.getForces():
        
        if "Nonbonded" in force.__class__.__name__:
            # NONBONDED
            charges = Quantity(param_dict["atom_q"], grappa_units.CHARGE_UNIT).value_in_unit(OPENMM_CHARGE_UNIT)
            sigmas = Quantity(param_dict["atom_sigma"], grappa_units.DISTANCE_UNIT).value_in_unit(OPENMM_SIGMA_UNIT)
            epsilons = Quantity(param_dict["atom_epsilon"], grappa_units.ENERGY_UNIT).value_in_unit(OPENMM_EPSILON_UNIT)

            for j in range(force.getNumParticles()):
                idx = np.where(param_dict["atom_idxs"] == j)[0]
                if len(idx) == 0: # if the atom is not in the param_dict,
                    continue
                if len(idx) > 1:
                    raise ValueError(f"Atom {j} appears multiple times in the param_dict.")
                idx = idx[0] # get the index of the atom in the param_dict
                charge, sigma, epsilon = charges[j], sigmas[j], epsilons[j]
                force.setParticleParameters(index=j, charge=charge, sigma=sigma, epsilon=epsilon)
        
        
        if "BondForce" in str(force.__class__):

            bonds = param_dict["bond_idxs"]
            bond_ks = Quantity(param_dict["bond_k"], grappa_units.FORCE_CONSTANT_UNIT).value_in_unit(OPENMM_BOND_K_UNIT)
            bond_eqs = Quantity(param_dict["bond_eq"], grappa_units.DISTANCE_UNIT).value_in_unit(OPENMM_BOND_EQ_UNIT)

            for i in range(force.getNumBonds()):
                idx1, idx2, _,_ = force.getBondParameters(i)
                idxs = np.array([idx1, idx2])
                indices = np.where(np.all(bonds == idxs, axis=1))[0]
                
                if len(indices) == 0: # only overwrite forces that are in the param_dict
                    continue
                if len(indices) > 1:
                    raise ValueError("Bond {} is defined multiple times in the param_dict.".format(idxs))
                idx = indices[0]
                k = bond_ks[idx]
                eq = bond_eqs[idx]
                force.setBondParameters(i, idx1, idx2, eq, k)

    # ANGLES
    angles = param_dict["angle_idxs"]
    angle_ks = Quantity(param_dict["angle_k"], grappa_units.ANGLE_FORCE_CONSTANT_UNIT).value_in_unit(OPENMM_ANGLE_K_UNIT)
    angle_eqs = Quantity(param_dict["angle_eq"], grappa_units.ANGLE_UNIT).value_in_unit(OPENMM_ANGLE_EQ_UNIT)

    angle_force = openmm.HarmonicAngleForce()
    for idxs, k, eq in zip(angles, angle_ks, angle_eqs):
        idx1, idx2, idx3 = idxs
        angle_force.addAngle(idx1, idx2, idx3, eq, k)
    sys.addForce(angle_force)

    # TORSIONS
    torsion_force = openmm.PeriodicTorsionForce()
    for torsion_type in ["proper", "improper"]:

        torsions = param_dict[f"{torsion_type}_idxs"]
        torsion_ks = Quantity(param_dict[f"{torsion_type}_ks"], grappa_units.ENERGY_UNIT).value_in_unit(OPENMM_TORSION_K_UNIT)
        torsion_phases = Quantity(param_dict[f"{torsion_type}_phases"], grappa_units.ANGLE_UNIT).value_in_unit(OPENMM_TORSION_PHASE_UNIT)
        torsion_periodicities = param_dict[f"{torsion_type}_ns"]
        
        for idxs, ks, ns, phases in zip(torsions, torsion_ks, torsion_periodicities, torsion_phases):
            idx1, idx2, idx3, idx4 = idxs

            for i, k in enumerate(ks):
                if k == 0.0:
                    continue

                phase = phases[i]
                n = ns[i]

                torsion_force.addTorsion(idx1, idx2, idx3, idx4, n, phase, k)
        
    sys.addForce(torsion_force)

    return sys