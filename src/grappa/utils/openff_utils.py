
import numpy as np
import pkgutil

from typing import Tuple, Set, Dict, Union, List


def get_sp_hybridization_encoding(openff_mol:"openff.toolkit.Molecule")->np.ndarray:
    """
    Returns a numpy array of shape (n_atoms, 6) that one-hot encodes wheter the atom can be described by a given hybridization type.
    """
    from rdkit.Chem.rdchem import HybridizationType

    # define the one hot encoding:
    hybridization_conversion = [
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
        HybridizationType.S,
    ]

    mol = openff_mol.to_rdkit()
    return np.array(
            [
                [
                    int(atom.GetHybridization() == hybridization) for hybridization in hybridization_conversion
                ]
                for atom in mol.GetAtoms()
            ]
        )

def get_is_aromatic(openff_mol:"openff.toolkit.Molecule")->np.ndarray:
    """
    Returns a numpy array of shape (n_atoms, 1) that one-hot encodes wheter the atom is aromatic.
    """
    mol = openff_mol.to_rdkit()
    return np.array(
            [
                [
                    int(atom.GetIsAromatic())
                ]
                for atom in mol.GetAtoms()
            ]
        )

def get_openmm_system(mapped_smiles:str, openff_forcefield:str='openff_unconstrained-1.2.0.offxml', partial_charges:Union[np.ndarray, list, int]=None, **system_kwargs)->Tuple["openmm.System", "openmm.Topology", "openff.toolkit.Molecule"]:
    """
    Returns system, topology, openff_molecule.
    Supported (tested) force fields:
    - gaff-2.11
    - openff-1.2.0.offxml
    - openff-2.0.0.offxml
    - openff_unconstrained-1.2.0.offxml
    - openff_unconstrained-2.0.0.offxml

    """
    from openff.toolkit import ForceField, Topology
    from openff.toolkit.topology import Molecule

    mol = Molecule.from_mapped_smiles(mapped_smiles, allow_undefined_stereo=True)
    topology = Topology.from_molecules(mol)
    openmm_topology = topology.to_openmm()

    if partial_charges is not None:
        if isinstance(partial_charges, np.ndarray):
            partial_charges = partial_charges.tolist()
        elif isinstance(partial_charges, int):
            partial_charges = [partial_charges] * len(mol.atoms)
        else:
            if not isinstance(partial_charges, list):
                raise TypeError(f"partial_charges must be either a list, a numpy array or an integer but is {type(partial_charges)}")

        # use the charges given in the raw molecule:
        from openff.units import unit as openff_unit
        mol.partial_charges = partial_charges * openff_unit.elementary_charge


    if 'openff' in openff_forcefield:
        ff = ForceField(openff_forcefield)

        if partial_charges is not None:
            # use the charges given in the raw molecule:
            openmm_system = ff.create_openmm_system(topology, charge_from_molecules=[mol], **system_kwargs)
        
        else:
            # calculate the charges from the force field (expensive!)
            openmm_system = ff.create_openmm_system(topology, **system_kwargs)

    elif 'gaff' in openff_forcefield:
        # assert that openmmforcefields is installed:
        assert pkgutil.find_loader("openmmforcefields") is not None, "openmmforcefields must be installed if non-openff force fields are being used."

        from openmmforcefields.generators import SystemGenerator
        top = mol.to_topology().to_openmm()

        if partial_charges is not None:
            raise NotImplementedError("Externally given partial charges are not supported yet for openmmforcefields force fields.")
        if not len(system_kwargs) == 0:
            raise NotImplementedError("Externally given system kwargs are not supported yet for openmmforcefields force fields.")

        system_generator = SystemGenerator(small_molecule_forcefield=openff_forcefield)

        openmm_system = system_generator.create_system(
            topology=top,
            molecules=mol,

        )
        # NOTE: what about given charges and system kwargs in this case?

    else:
        raise NotImplementedError("Only openff and gaff force fields are supported.")
    
    return openmm_system, openmm_topology, mol