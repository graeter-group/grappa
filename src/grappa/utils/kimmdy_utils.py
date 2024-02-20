# import pkgutil

# if pkgutil.find_spec('kimmdy'):

#     import logging
#     import numpy as np

#     import math
#     from typing import Union

#     from kimmdy.topology.topology import Topology
#     from kimmdy.topology.atomic import Atom, Bond, Angle, Dihedral, MultipleDihedrals
#     from kimmdy.plugins import Parameterizer

#     from grappa.data import Molecule
#     from grappa.data import Parameters

#     from grappa.utils.loading_utils import model_from_tag
#     from grappa.grappa import Grappa

#     from openmm import unit as openmm_unit
#     from grappa.units import convert


#     # helper functions
#     def check_equal_length(d: dict, name: str):
#         lengths = [len(y) for y in d.values()]
#         assert (
#             len(set(lengths)) == 1
#         ), f"Different length of {name} parameters: { {k:len(v) for k, v in d.items()} }"


#     # unused?
#     def convert_to_python_types(array: Union[list, np.ndarray]) -> list:
#         return getattr(array, "tolist", lambda: array)()


#     def order_proper(idxs: np.ndarray) -> np.ndarray:
#         # center atoms of dihedral must have ascending value
#         # idx_list is array(array(list(i)),array(list(j)),array(list(k)),array(list(l)))
#         if idxs[1] < idxs[2]:
#             return idxs
#         else:
#             return np.flip(idxs)
        

#     # workflow functions
#     def build_molecule(top: Topology) -> Molecule:
#         at_map = top.ff.atomtypes
#         atom_info = {
#             "nr": [],
#             "atomic_number": [],
#             "partial_charges": [],
#             "sigma": [],
#             "epsilon": [],
#             "is_radical": [],
#         }
#         for atom in top.atoms.values():
#             atom_info["nr"].append(int(atom.nr))
#             atom_info["atomic_number"].append(int(at_map[atom.type].at_num))
#             atom_info["partial_charges"].append(float(atom.charge))
#             atom_info["sigma"].append(float(at_map[atom.type].sigma))
#             atom_info["epsilon"].append(float(at_map[atom.type].epsilon))
#             atom_info["is_radical"].append(int(atom.is_radical))

#         bonds = [(int(bond.ai), int(bond.aj)) for bond in top.bonds.values()]
#         impropers = [
#             (int(improper.ai), int(improper.aj), int(improper.ak), int(improper.al))
#             for improper in top.improper_dihedrals.values()
#         ]

#         mol = Molecule(
#             atoms=atom_info["nr"],
#             bonds=bonds,
#             impropers=impropers,
#             atomic_numbers=atom_info["atomic_number"],
#             partial_charges=atom_info["partial_charges"],
#             additional_features={
#                 k: np.asarray(v)
#                 for k, v in atom_info.items()
#                 if k not in ["nr", "atomic_number", "partial_charges"]
#             },
#             charge_model="classical",
#         )
#         return mol


#     def convert_parameters(parameters: Parameters) -> Parameters:
#         """Converts parameters to gromacs units
#         Assumes input parameters to be in  kcal/mol, Angstrom und rad
#         Gromac units mostly kJ/mol, nm and degree
#         """

#         distance_factor = convert(1, openmm_unit.angstrom, openmm_unit.nanometer)
#         degree_factor = convert(1, openmm_unit.radian, openmm_unit.degree)
#         energy_factor = convert(
#             1, openmm_unit.kilocalorie_per_mole, openmm_unit.kilojoule_per_mole
#         )

#         # convert parameters
#         parameters.bond_eq = parameters.bond_eq * distance_factor
#         parameters.bond_k = parameters.bond_k * energy_factor / np.power(distance_factor, 2)
#         # angles are given in degrees and force constants in kJ/mol/rad**2.
#         parameters.angle_eq = parameters.angle_eq * degree_factor
#         parameters.angle_k = parameters.angle_k * energy_factor

#         parameters.propers = np.array([order_proper(x) for x in parameters.propers])
#         parameters.proper_phases = parameters.proper_phases * degree_factor
#         parameters.proper_ks = parameters.proper_ks * energy_factor

#         parameters.improper_phases = parameters.improper_phases * degree_factor
#         parameters.improper_ks = parameters.improper_ks * energy_factor

#         # convert to list of strings
#         for k in parameters.__annotations__.keys():
#             v = getattr(parameters, k)
#             if len(v) == 0:
#                 logging.warning(f"Parameter list {k} is empty.")
#             else:
#                 if isinstance(v[0], float):
#                     v_list = [f"{i:11.4f}".strip() for i in v]
#                 elif isinstance(v[0], np.ndarray) and isinstance(v[0, 0], float):
#                     v_list = []
#                     for sub_list in v:
#                         v_list.append([f"{i:11.4f}".strip() for i in sub_list])
#                 else:
#                     v_list = v.astype(str).tolist()
#                 setattr(parameters, k, v_list)

#         return parameters


#     def apply_parameters(top: Topology, parameters: Parameters):
#         """Applies parameters to topology

#         parameter structure is defined in grappa.data.Parameters.Parameters
#         assume units are according to https://manual.gromacs.org/current/reference-manual/definitions.html
#         namely: length [nm], mass [kg], time [ps], energy [kJ/mol], force [kJ mol-1 nm-1], angle [deg]
#         """

#         ## atoms
#         # Nothing to do here because partial charges are dealt with elsewhere

#         ## bonds
#         for i, idx in enumerate(parameters.bonds):
#             tup = tuple(idx)
#             if not top.bonds.get(tup):
#                 # raise KeyError(f"bad index {tup} in {list(top.bonds.keys())}")
#                 logging.warning(f"Ignored parameters with invalid ids: {tup} for bonds")
#                 continue
#             top.bonds[tup] = Bond(
#                 *tup,
#                 funct="1",
#                 c0=parameters.bond_eq[i],
#                 c1=parameters.bond_k[i],
#             )

#         ## angles
#         for i, idx in enumerate(parameters.angles):
#             tup = tuple(idx)
#             if not top.angles.get(tup):
#                 # raise KeyError(f"bad index {tup} in {list(top.angles.keys())}")
#                 logging.warning(f"Ignored parameters with invalid ids: {tup} for angles")
#                 continue
#             top.angles[tup] = Angle(
#                 *tup,
#                 funct="1",
#                 c0=parameters.angle_eq[i],
#                 c1=parameters.angle_k[i],
#             )

#         ## proper dihedrals
#         for i, idx in enumerate(parameters.propers):
#             tup = tuple(idx)
#             if not top.proper_dihedrals.get(tup):
#                 # raise KeyError(f"bad index {tup} in {list(top.proper_dihedrals.keys())}")
#                 logging.warning(
#                     f"Ignored parameters with invalid ids: {tup} for proper dihedrals"
#                 )
#                 continue
#             dihedral_dict = {}
#             for ii in range(len(parameters.proper_ks[i])):
#                 n = str(ii + 1)
#                 dihedral_dict[n] = Dihedral(
#                     *tup,
#                     funct="9",
#                     c0=parameters.proper_phases[i][ii],
#                     c1=parameters.proper_ks[i][ii],
#                     periodicity=n,
#                 )
#             top.proper_dihedrals[tup] = MultipleDihedrals(
#                 *tup, funct="9", dihedrals=dihedral_dict
#             )

#         ## improper dihedrals
#         top.improper_dihedrals = {}
#         for i, idx in enumerate(parameters.impropers):
#             tup = tuple(idx)
#             for ii in range(len(parameters.improper_ks[i])):
#                 n = str(ii + 1)
#                 if not math.isclose(
#                     float(parameters.improper_ks[i][ii]), 0.0, abs_tol=1e-3
#                 ):
#                     if curr_improper := (top.improper_dihedrals.get(tup)) is None:
#                         top.improper_dihedrals[tup] = Dihedral(
#                             *tup,
#                             funct="4",
#                             c0=parameters.improper_phases[i][ii],
#                             c1=parameters.improper_ks[i][ii],
#                             periodicity=n,
#                         )
#                     else:
#                         new_improper = Dihedral(
#                             *tup,
#                             funct="4",
#                             c0=parameters.improper_phases[i][ii],
#                             c1=parameters.improper_ks[i][ii],
#                             periodicity=n,
#                         )
#                         if new_improper.c1 > curr_improper.c1:
#                             top.improper_dihedrals[tup] = new_improper
#                             deserted_improper = curr_improper
#                         else:
#                             deserted_improper = new_improper

#                         logging.warning(
#                             f"There are multiple improper dihedrals for {tup} and only one can be chosen, dihedral p{deserted_improper} will be ignored."
#                         )

#         return

#     def get_kimmdy_parametrizer(grappa_model:Grappa, device:str="cpu") -> Parameterizer:

#     class GrappaInterface(Parameterizer):
#         def parameterize_topology(
#             self, current_topology: Topology, focus_nr: list[str] = []
#         ) -> Topology:
#             ## get atoms, bonds, radicals in required format
#             mol = build_molecule(current_topology)

#             model = load_model()

#             # initialize class that handles ML part
#             grappa = Grappa(model, device="cpu")
#             parameters = grappa.predict(mol)

#             # convert units et cetera
#             parameters = convert_parameters(parameters)

#             # apply parameters
#             apply_parameters(current_topology, parameters)
#             return current_topology