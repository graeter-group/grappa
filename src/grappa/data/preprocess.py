"""
Provides methods to create a grappa dataset from npz files.
"""

import numpy as np
import openmm
from datatypes import RawMolecule, GrappaMolecule
import pkgutil


def preprocess(dspath:str, classical_forcefield:str, targetpath:str=None, calc_energies:bool=False, openff_forcefield:bool=False, partial_charge_key:str=None, checks:bool=False) -> None:
    """
    Preprocess a whole dataset stored as npz files that define a RawMolecule.
    """

def preprocess_molecule(raw_molecule:RawMolecule, classical_forcefield:str, calc_energies:bool=False, openff_forcefield:bool=False, partial_charge_key:str=None, checks:bool=False) -> GrappaMolecule:
    """
    Preprocess a single molecule with a single classical force field. The forcefield must either be an openff force field or a force field supported by openmm directly. openmmforcefields are counted as openff force fields since they require an openff molecule.

    openff_forcefield: If True, the classical_forcefield is assumed to be an openff force field.
    partial_charge_key: If not None, the partial charges are taken from the raw_molecule with the given key and not calculated from the classical force field.
    """

    system_kwargs = {} #NOTE

    # load partial charges if given:
    partial_charges = None
    if partial_charge_key is not None:
        assert partial_charge_key in raw_molecule.keys(), "The raw molecule must have a key '{}' if partial_charge_key is not None.".format(partial_charge_key)
        partial_charges = raw_molecule[partial_charge_key]
        assert partial_charges.shape == raw_molecule['atomic_numbers'].shape, "The partial charges must have the same shape as the atomic numbers."
        assert partial_charges.ndim == 1, "The partial charges must be 1D."
    

    # obtain an openmm system
    ############################
    if openff_forcefield:
        assert 'mapped_smiles' in raw_molecule.keys(), "The raw molecule must have a 'mapped_smiles' key if openff_forcefield is True."
        assert pkgutil.find_loader("openff.toolkit") is not None, "openff.toolkit must be installed if openff_forcefield is True."

        from openff.toolkit import ForceField, Topology
        from openff.toolkit.topology import Molecule

        if 'openff' in classical_forcefield:
            mol = Molecule.from_mapped_smiles(raw_molecule['mapped_smiles'], allow_undefined_stereo=True)
            topology = Topology.from_molecules(mol)
            ff = ForceField(classical_forcefield)

            if partial_charges is not None and not calc_energies:
                # use the charges given in the raw molecule:
                from openff.units.unit import elementary_charge                
                mol.partial_charges = partial_charges * elementary_charge
                openmm_system = ff.create_openmm_system(topology, charge_from_molecules=[mol], **system_kwargs)
            
            else:
                # calculate the charges from the force field (expensive!)
                openmm_system = ff.create_openmm_system(topology, **system_kwargs)


        else:
            # assert that openmmforcefields is installed:
            assert pkgutil.find_loader("openmmforcefields") is not None, "openmmforcefields must be installed if non-openff force fields are being used."

            from openmmforcefields.generators import SystemGenerator
            top = mol.to_topology().to_openmm()

            system_generator = SystemGenerator(small_molecule_forcefield=classical_forcefield)

            openmm_system = system_generator.create_system(
                topology=top,
                molecules=mol,

            )
            # NOTE: what about given charges and system kwargs?

    else:
        ff = openmm.app.ForceField(classical_forcefield)
        openmm_system = ff.createSystem(topology)
        

    if partial_charges is None:
        # obtain the charges from the openmm system:
        pass
    
    if calc_energies:
        # calculate the nonbonded and total energies of the classical force field:
        pass

    # obtain the idxs of bonds, angles, torsions and improper torsions:
    pass


def get_interaction_tuples(openmm_system, check=False):
    # create bond, angle, proper with rdkit
    # use the impropers from openmm by using the is_imporper function
    # if check is true, then the interaction tuples are checked for consistency with the openmm system:
    # all openmm interactions must be present in the corresponding interaction tuples
    pass