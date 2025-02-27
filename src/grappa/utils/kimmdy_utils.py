import pkgutil
import importlib

"""
This is used for the gromacs wrapper.
"""

if importlib.util.find_spec('kimmdy') is not None:
    from kimmdy.topology.topology import Topology
    from kimmdy.topology.atomic import Atom, Bond, Angle, Dihedral, MultipleDihedrals
    from kimmdy.plugins import Parameterizer

import logging
import numpy as np

import math
from typing import Union, Optional, Set, List, Tuple


from grappa.data import Molecule
from grappa.data import Parameters

from grappa.grappa import Grappa

from grappa.units import Unit, Quantity
from grappa import units
from grappa.constants import GrappaUnits

# define the units that gromacs uses. Toghether with grappa.constants.GrappaUnits, this defines how we will convert the output of the ML model.
# https://manual.gromacs.org/current/reference-manual/definitions.html
# namely: length [nm], mass [kg], time [ps], energy [kJ/mol], force [kJ mol-1 nm-1], angle [deg]
# (in Gromacs, angles are given in degrees and angle force constants in kJ/mol/rad**2...)
GROMACS_BOND_EQ = units.nanometer
GROMACS_BOND_K = units.kilojoule_per_mole / units.nanometer ** 2
GROMACS_ANGLE_EQ = units.degree
GROMACS_ANGLE_K = units.kilojoule_per_mole / units.radian ** 2
GROMACS_TORSION_PHASE = units.degree
GROMACS_TORSION_K = units.kilojoule_per_mole



# helper functions
def check_equal_length(d: dict, name: str):
    lengths = [len(y) for y in d.values()]
    assert (
        len(set(lengths)) == 1
    ), f"Different length of {name} parameters: { {k:len(v) for k, v in d.items()} }"



def order_proper(idxs: np.ndarray) -> np.ndarray:
    # center atoms of dihedral must have ascending value
    # idx_list is array(array(list(i)),array(list(j)),array(list(k)),array(list(l)))
    if idxs[1] < idxs[2]:
        return idxs
    else:
        return np.flip(idxs)
    

# workflow functions
def build_molecule(top: Topology, build_nrs:Set[str], charge_model:str='amber99') -> Molecule:
    '''
    Build a grappa.data.Molecule from a kimmdy.topology.topology.Topology

    - top: Topology to be represented by the Molecule
    - charge_model: tag that describes where the partial charges in the topology will come from. Possible values:
        - 'amber99': the charges are assigned using a classical force field. For grappa-1.1, this is only possible for peptides and proteins, where classical refers to the charges from the amber99sbildn force field.
        - 'am1BCC': the charges are assigned using the am1bcc method. These charges need to be used for rna and small molecules in grappa-1.1.
    '''
    at_map = top.ff.atomtypes
    atom_info = {
        "nr": [],
        "atomic_number": [],
        "partial_charges": [],
        "sigma": [],
        "epsilon": [],
    }
    for atom in top.atoms.values():
        if atom.nr in build_nrs:
            atom_info["nr"].append(int(atom.nr))
            atom_info["atomic_number"].append(int(at_map[atom.type].at_num))
            atom_info["partial_charges"].append(float(atom.charge))
            atom_info["sigma"].append(float(at_map[atom.type].sigma))
            atom_info["epsilon"].append(float(at_map[atom.type].epsilon))

    bonds = [(int(bond.ai), int(bond.aj)) for bond in top.bonds.values() if all(atom_nr in build_nrs for atom_nr in [bond.ai,bond.aj])]
    impropers = [
        (int(improper.ai), int(improper.aj), int(improper.ak), int(improper.al))
        for improper in top.improper_dihedrals.values() if all(atom_nr in build_nrs for atom_nr in [improper.ai,improper.aj, improper.ak,improper.al])
    ]

    mol = Molecule(
        atoms=atom_info["nr"],
        bonds=bonds,
        impropers=impropers,
        atomic_numbers=atom_info["atomic_number"],
        partial_charges=atom_info["partial_charges"],
        additional_features={
            k: np.asarray(v)
            for k, v in atom_info.items()
            if k not in ["nr", "atomic_number", "partial_charges"]
        },
        charge_model=charge_model,
    )
    return mol


def convert_parameters(parameters: Parameters) -> Parameters:
    """Converts parameters to gromacs units
    Assumes input parameters to be in  kcal/mol, Angstrom und rad
    Gromac units mostly kJ/mol, nm and degree
    """

    # convert parameters to the right units
    parameters.bond_eq = Quantity(parameters.bond_eq, GrappaUnits.BOND_EQ).value_in_unit(GROMACS_BOND_EQ)
    parameters.bond_k = Quantity(parameters.bond_k, GrappaUnits.BOND_K).value_in_unit(GROMACS_BOND_K)
    # angles are given in degrees and force constants in kJ/mol/rad**2.
    parameters.angle_eq = Quantity(parameters.angle_eq, GrappaUnits.ANGLE_EQ).value_in_unit(GROMACS_ANGLE_EQ)
    parameters.angle_k = Quantity(parameters.angle_k, GrappaUnits.ANGLE_K).value_in_unit(GROMACS_ANGLE_K)

    parameters.propers = np.array([order_proper(x) for x in parameters.propers])
    parameters.proper_phases = Quantity(parameters.proper_phases, GrappaUnits.TORSION_PHASE).value_in_unit(GROMACS_TORSION_PHASE)
    parameters.proper_ks = Quantity(parameters.proper_ks, GrappaUnits.TORSION_K).value_in_unit(GROMACS_TORSION_K)

    parameters.improper_phases = Quantity(parameters.improper_phases, GrappaUnits.TORSION_PHASE).value_in_unit(GROMACS_TORSION_PHASE)
    parameters.improper_ks = Quantity(parameters.improper_ks, GrappaUnits.TORSION_K).value_in_unit(GROMACS_TORSION_K)


    # convert to list of strings
    for k in parameters.__annotations__.keys():
        v = getattr(parameters, k)
        if len(v) == 0:
            logging.warning(f"Parameter list {k} is empty.")
        else:
            if isinstance(v[0], float):
                v_list = [f"{i:11.4f}".strip() for i in v]
            elif isinstance(v[0], np.ndarray) and isinstance(v[0, 0], float):
                v_list = []
                for sub_list in v:
                    v_list.append([f"{i:11.4f}".strip() for i in sub_list])
            else:
                v_list = v.astype(str).tolist()
            setattr(parameters, k, v_list)

    return parameters


def apply_parameters(top: 'Topology', parameters: Parameters, apply_nrs: Set[str]):
    """
    Applies parameters to topology. (writes the parameters in the topology)
    To that end, we check whether equivalent tuple orderings 

    parameter structure is defined in grappa.data.Parameters.Parameters
    assume units are according to https://manual.gromacs.org/current/reference-manual/definitions.html
    namely: length [nm], mass [kg], time [ps], energy [kJ/mol], force [kJ mol-1 nm-1], angle [deg]
    """

    ## atoms
    # Nothing to do here because partial charges are dealt with elsewhere

    ## bonds
    for i, idx in enumerate(parameters.bonds):
        if all(atom_nr in apply_nrs for atom_nr in idx):
            tup = tuple(idx)
            tup = find_bond(tup, top)
            if not tup:
                raise ValueError(f"Invalid bond tuple {tup}")

            top.bonds[tup] = Bond(
                *tup,
                funct="1",
                c0=parameters.bond_eq[i],
                c1=parameters.bond_k[i],
            )

    ## angles
    for i, idx in enumerate(parameters.angles):
        if all(atom_nr in apply_nrs for atom_nr in idx):
            tup = tuple(idx)
            tup = find_angle(tup, top)
            if not tup:
                raise ValueError(f"Invalid angle tuple {tup}")

            top.angles[tup] = Angle(
                *tup,
                funct="1",
                c0=parameters.angle_eq[i],
                c1=parameters.angle_k[i],
            )

    ## proper dihedrals
    for i, idx in enumerate(parameters.propers):
        if all(atom_nr in apply_nrs for atom_nr in idx):
            tup = tuple(idx)

            # find the proper dihedral tuple in the topology that is equivalent to the given tuple
            tup = find_proper(tup, top)
            if not tup:
                raise ValueError(f"Invalid proper dihedral tuple {tup}")

            dihedral_dict = {}
            for ii in range(len(parameters.proper_ks[i])):
                n = str(ii + 1)
                dihedral_dict[n] = Dihedral(
                    *tup,
                    funct="9",
                    c0=parameters.proper_phases[i][ii],
                    c1=parameters.proper_ks[i][ii],
                    periodicity=n,
                )
            top.proper_dihedrals[tup] = MultipleDihedrals(
                *tup, funct="9", dihedrals=dihedral_dict
            )

    ## improper dihedrals
    # clear old dihedrals for the apply_nrs region
    for improper in list(top.improper_dihedrals.values()):
        tup = tuple([improper.ai,improper.aj,improper.ak,improper.al])
        if all(atom_nr in apply_nrs for atom_nr in tup):
            top.improper_dihedrals.pop(tup)

    for i, idx in enumerate(parameters.impropers):
        if all(atom_nr in apply_nrs for atom_nr in idx):
            tup = tuple(idx)
            dihedral_dict = {}
            for ii in range(len(parameters.improper_ks[i])):
                n = str(ii + 1)
                dihedral_dict[n] = Dihedral(
                    *tup,
                    funct="4",
                    c0=parameters.improper_phases[i][ii],
                    c1=parameters.improper_ks[i][ii],
                    periodicity=n,
                )
            top.improper_dihedrals[tup] = MultipleDihedrals(
                *tup, funct="4", dihedrals=dihedral_dict
            )

    return


class KimmdyGrappaParameterizer('Parameterizer'):
    '''
    Kimmdy Parameterizer that uses a Grappa model to parameterize a Topology.

    - grappa_instance: Grappa instance to use for parameterization
    - charge_model: tag that describes where the partial charges in the topology will come from. Possible values:
        - 'amber99': the charges are assigned using a classical force field. For grappa-1.1, this is only possible for peptides and proteins, where classical refers to the charges from the amber99sbildn force field.
        - 'am1BCC': the charges are assigned using the am1bcc method. These charges need to be used for rna and small molecules in grappa-1.1.
    '''
    def __init__(self, *args, grappa_instance: Grappa, charge_model:str='amber99', plot_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grappa_instance = grappa_instance
        self.field_of_view = grappa_instance.field_of_view
        self.charge_model = charge_model
        self.plot_path = plot_path

    def parameterize_topology(
        self, current_topology: 'Topology', focus_nrs:  Optional[Set[str]] = None
    ) -> 'Topology':
        
        if not focus_nrs:
            print(f"Parameterizing molecule without focus")
            build_nrs = set([atom.nr for atom in current_topology.atoms.values()])
            apply_nrs = build_nrs
        else:
            # field of view relates to attention layers and convolutions; + 3 to get dihedrals and ring membership (up to 6 membered rings, for larger rings this should be higher)
            field_of_view = self.field_of_view
            # new parameters are applied to atoms within the field of view of the focus_nrs atoms. 
            # For all of those to have the same field of view as in the whole molecule, another field of view is added for building the molecule
            apply_nrs = current_topology.get_neighbors(focus_nrs,field_of_view)
            build_nrs = current_topology.get_neighbors(apply_nrs,field_of_view)


        ## get atoms, bonds, radicals in required format
        mol = build_molecule(current_topology, build_nrs, charge_model=self.charge_model)

        parameters = self.grappa_instance.predict(mol)

        # create a plot for visual inspection:
        if self.plot_path is not None:
            parameters.plot(filename=str(self.plot_path))

        # convert units et cetera
        parameters = convert_parameters(parameters)

        # apply parameters
        apply_parameters(current_topology, parameters, build_nrs)
        return current_topology
        
def find_angle(tup: tuple, top: 'Topology') -> Tuple[int, int, int]:
    """
    Find the angle tuple in the topology that is equivalent to the given tuple.
    If no equivalent tuple is found, it logs a warning and returns False.
    """
    if not top.angles.get(tup):
        # try equivalent tuples using the symmetries of the angle:
        # angle_{ijk} = angle_{kji} (reversal)
        equivalent_tups = [
            tuple(reversed(tup))
        ]
        
        tups_in_topology = [tup_eq for tup_eq in equivalent_tups if top.angles.get(tup_eq)]

        if len(tups_in_topology) == 0:
            logging.warning(
                f"Ignored parameters with invalid ids: {tup} for angles"
            )
            return None
        elif len(tups_in_topology) > 1:
            logging.warning(
                f"Multiple equivalent tuples found for {tup} in angles"
            )
        else:
            tup = tups_in_topology[0]
    return tup

def find_bond(tup: tuple, top: 'Topology') -> Tuple[int, int]:
    """
    Find the bond tuple in the topology that is equivalent to the given tuple.
    If no equivalent tuple is found, it logs a warning and returns False.
    """
    if not top.bonds.get(tup):
        # try equivalent tuples using the symmetries of the bond:
        # bond_{ij} = bond_{ji} (reversal)
        equivalent_tups = [
            tuple(reversed(tup))
        ]
        
        tups_in_topology = [tup_eq for tup_eq in equivalent_tups if top.bonds.get(tup_eq)]

        if len(tups_in_topology) == 0:
            logging.warning(
                f"Ignored parameters with invalid ids: {tup} for bonds"
            )
            return None
        elif len(tups_in_topology) > 1:
            logging.warning(
                f"Multiple equivalent tuples found for {tup} in bonds"
            )
        else:
            tup = tups_in_topology[0]
    return tup


def find_proper(tup: tuple, top: 'Topology') -> Tuple[int, int, int, int]:
    """
    Find the proper dihedral tuple in the topology that is equivalent to the given tuple.
    If no equivalent tuple is found, it logs a warning and returns False.
    """

    if not top.proper_dihedrals.get(tup):
        # try equivalent tuples. use the symmetries of the dihedral angle (see Grappa paper appendix):
        # cos(phi_{ijkl}) = cos(phi_{lkji}) (reversal)
        # = cos(phi_{ljki}) = cos(phi_{ikjl}) (permutation of the outer atoms)
        equivalent_tups = [
            tuple(reversed(tup)),
            (tup[3], tup[1], tup[2], tup[0]),
            (tup[0], tup[2], tup[1], tup[3]),
        ]
        
        tups_in_topology = list([tup_eq for tup_eq in equivalent_tups if top.proper_dihedrals.get(tup_eq)])

        if len(tups_in_topology) == 0:
            logging.warning(
                f"Ignored parameters with invalid ids: {tup} for proper dihedrals"
            )
            return None
        elif len(tups_in_topology) > 1:
            logging.warning(
                f"Multiple equivalent tuples found for {tup} in proper dihedrals"
            )
        else:
            tup = tups_in_topology[0]
    return tup