import pkgutil
import importlib

"""
This is used for the gromacs wrapper.
"""

if importlib.util.find_spec('kimmdy') is not None:

    import logging
    import numpy as np

    import math
    from typing import Union

    from kimmdy.topology.topology import Topology
    from kimmdy.topology.atomic import Atom, Bond, Angle, Dihedral, MultipleDihedrals
    from kimmdy.plugins import Parameterizer

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


    # unused?
    def convert_to_python_types(array: Union[list, np.ndarray]) -> list:
        return getattr(array, "tolist", lambda: array)()


    def order_proper(idxs: np.ndarray) -> np.ndarray:
        # center atoms of dihedral must have ascending value
        # idx_list is array(array(list(i)),array(list(j)),array(list(k)),array(list(l)))
        if idxs[1] < idxs[2]:
            return idxs
        else:
            return np.flip(idxs)
        

    # workflow functions
    def build_molecule(top: Topology, charge_model:str='classical') -> Molecule:
        '''
        Build a grappa.data.Molecule from a kimmdy.topology.topology.Topology

        - top: Topology to be represented by the Molecule
        - charge_model: tag that describes where the partial charges in the topology will come from. Possible values:
            - 'classical': the charges are assigned using a classical force field. For grappa-1.1, this is only possible for peptides and proteins, where classical refers to the charges from the amber99sbildn force field.
            - 'am1BCC': the charges are assigned using the am1bcc method. These charges need to be used for rna and small molecules in grappa-1.1.
        '''
        at_map = top.ff.atomtypes
        atom_info = {
            "nr": [],
            "atomic_number": [],
            "partial_charges": [],
            "sigma": [],
            "epsilon": [],
            "is_radical": [], # this feature is not used anymore in grappa-1.1!
        }
        for atom in top.atoms.values():
            atom_info["nr"].append(int(atom.nr))
            atom_info["atomic_number"].append(int(at_map[atom.type].at_num))
            atom_info["partial_charges"].append(float(atom.charge))
            atom_info["sigma"].append(float(at_map[atom.type].sigma))
            atom_info["epsilon"].append(float(at_map[atom.type].epsilon))
            atom_info["is_radical"].append(int(atom.is_radical))

        bonds = [(int(bond.ai), int(bond.aj)) for bond in top.bonds.values()]
        impropers = [
            (int(improper.ai), int(improper.aj), int(improper.ak), int(improper.al))
            for improper in top.improper_dihedrals.values()
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


    def apply_parameters(top: Topology, parameters: Parameters):
        """Applies parameters to topology

        parameter structure is defined in grappa.data.Parameters.Parameters
        assume units are according to https://manual.gromacs.org/current/reference-manual/definitions.html
        namely: length [nm], mass [kg], time [ps], energy [kJ/mol], force [kJ mol-1 nm-1], angle [deg]
        """

        ## atoms
        # Nothing to do here because partial charges are dealt with elsewhere

        ## bonds
        for i, idx in enumerate(parameters.bonds):
            tup = tuple(idx)
            if not top.bonds.get(tup):
                # raise KeyError(f"bad index {tup} in {list(top.bonds.keys())}")
                logging.warning(f"Ignored parameters with invalid ids: {tup} for bonds")
                continue
            top.bonds[tup] = Bond(
                *tup,
                funct="1",
                c0=parameters.bond_eq[i],
                c1=parameters.bond_k[i],
            )

        ## angles
        for i, idx in enumerate(parameters.angles):
            tup = tuple(idx)
            if not top.angles.get(tup):
                # raise KeyError(f"bad index {tup} in {list(top.angles.keys())}")
                logging.warning(f"Ignored parameters with invalid ids: {tup} for angles")
                continue
            top.angles[tup] = Angle(
                *tup,
                funct="1",
                c0=parameters.angle_eq[i],
                c1=parameters.angle_k[i],
            )

        ## proper dihedrals
        for i, idx in enumerate(parameters.propers):
            tup = tuple(idx)
            if not top.proper_dihedrals.get(tup):
                # raise KeyError(f"bad index {tup} in {list(top.proper_dihedrals.keys())}")
                logging.warning(
                    f"Ignored parameters with invalid ids: {tup} for proper dihedrals"
                )
                continue
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
        top.improper_dihedrals = {}
        for i, idx in enumerate(parameters.impropers):
            tup = tuple(idx)
            for ii in range(len(parameters.improper_ks[i])):
                n = str(ii + 1)
                if not math.isclose(
                    float(parameters.improper_ks[i][ii]), 0.0, abs_tol=1e-6
                ):
                    curr_improper = top.improper_dihedrals.get(tup)
                    if curr_improper is None:
                        top.improper_dihedrals[tup] = Dihedral(
                            *tup,
                            funct="4",
                            c0=parameters.improper_phases[i][ii],
                            c1=parameters.improper_ks[i][ii],
                            periodicity=n,
                        )
                    else:
                        new_improper = Dihedral(
                            *tup,
                            funct="4",
                            c0=parameters.improper_phases[i][ii],
                            c1=parameters.improper_ks[i][ii],
                            periodicity=n,
                        )
                        if new_improper.c1 > curr_improper.c1:
                            top.improper_dihedrals[tup] = new_improper
                            deserted_improper = curr_improper
                        else:
                            deserted_improper = new_improper

                        logging.warning(
                            f"There are multiple improper dihedrals for {tup} and only one can be chosen, dihedral p{deserted_improper} will be ignored."
                        )

        return


    class KimmdyGrappaParameterizer(Parameterizer):
        '''
        Kimmdy Parameterizer that uses a Grappa model to parameterize a Topology.

        - grappa_model: Grappa model to use for parameterization
        - charge_model: tag that describes where the partial charges in the topology will come from. Possible values:
            - 'classical': the charges are assigned using a classical force field. For grappa-1.1, this is only possible for peptides and proteins, where classical refers to the charges from the amber99sbildn force field.
            - 'am1BCC': the charges are assigned using the am1bcc method. These charges need to be used for rna and small molecules in grappa-1.1.
        '''
        def __init__(self, *args, grappa_model: Grappa, charge_model:str='classical', **kwargs):
            super().__init__(*args, **kwargs)
            self.grappa_model = grappa_model
            self.charge_model = charge_model

        def parameterize_topology(
            self, current_topology: Topology, focus_nr: list[str] = []
        ) -> Topology:
            
            ## get atoms, bonds, radicals in required format
            mol = build_molecule(current_topology, charge_model=self.charge_model)

            parameters = self.grappa_model.predict(mol)

            # convert units et cetera
            parameters = convert_parameters(parameters)

            # apply parameters
            apply_parameters(current_topology, parameters)
            return current_topology