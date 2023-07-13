"""
Class wrapping a model and providing methods for translating various input types to dgl graphs whcih can be processed by the model and translate back to various output types.
"""

# MAKE MAX ELEMENT CLASS MEMBER!

import torch
import dgl
from typing import Union, List, Tuple, Dict, Callable
from .ff_utils.create_graph.utils import process_input, process_output
from .deploy.deploy import model_from_path
import openmm.app.topology
import openmm.app
from pathlib import Path
import numpy as np
import json

from .ff_utils.sysWriter import sysWriter

from .ff_utils.charge_models import model_from_dict

from openmm import unit

class ForceField:
    def __init__(self, model:Callable=None, model_path:Union[str, Path]=None, classical_ff=openmm.app.ForceField("amber99sbildn.xml"), charge_model:Callable=None) -> None:
        """
        Class wrapping a model and providing methods for translating various input types to dgl graphs whcih can be processed by the model and translate back to various output types.
        model_path: a path to a folder with a single .pt file containing a model-state_dict and a config.yaml file containing model hyperparameters for construction.
        model: a callable taking and returning a dgl graph.
        classical_ff: an openmm forcefield object used for nonbonded parameters and system initialization.
        units: a dictionary containing the openmm.unit used for the output dictionary.
            The keys and default values are:
            {
                "distance": unit.nanometer,
                "energy": unit.kilojoule_per_mole,
                "angle": unit.radian,
                "charge": unit.elementary_charge,
                "mass": unit.dalton,
            }
            Note that the derived parameters then are:
                FORCE_UNIT = ENERGY_UNIT / DISTANCE_UNIT
                FORCE_CONSTANT_UNIT = ENERGY_UNIT / (DISTANCE_UNIT ** 2)
                ANGLE_FORCE_CONSTANT_UNIT = ENERGY_UNIT / (ANGLE_UNIT ** 2)
                TORSION_FORCE_CONSTANT_UNIT = ENERGY_UNIT

        """
        if model is not None and model_path is not None:
            raise ValueError("Either model or model_path must be given, not both.")
        if model is None and model_path is None:
            raise ValueError("Either model or model_path must be given.")

        if model is not None:
            self.model = model
        else:
            self.model = model_from_path(model_path)
        
        self.classical_ff = classical_ff

        self.units = {
            "distance": unit.nanometer,
            "energy": unit.kilojoule_per_mole,
            "angle": unit.radian,
            "charge": unit.elementary_charge,
            "mass": unit.dalton,
        }

        self.use_improper = True # if False, does not use impropers, which allows for the prediction of parameters without the need of a classical forcefield. The reason is that improper ordering is not unique



    def set_charge_model(self, charge_model:Union[Callable, str])->None:
        """
        Sets the charge model to the given callable. May also be a tag.
        """
        if isinstance(charge_model, str):
            self.charge_model = model_from_dict(tag=charge_model)
        else:
            self.charge_model = charge_model


    def createSystem(self, top:openmm.app.topology.Topology, allow_radicals:False, **system_kwargs):

        writer = sysWriter(top, allow_radicals=allow_radicals, **system_kwargs)
        writer.init_graph(with_parameters=False)
        writer.forward_pass(model=self.model)
        writer.update_system()
    
        return writer.system
    

    def params_from_topology_dict(self, top_dict:Dict)->Dict:
        """
        Accepts a dictionary with the keys 'atoms', 'bonds' and 'radicals'. The atoms entry must be a list containing lists of the form:
        [atom_idx:int, residue_name:str, atom_name:str, residue_idx:int, [sigma:float, epsilon:float], atomic_number:int]
        e.g. [1, "CH3", "ACE", 1, [0.339967, 0.45773], 6]
        If the use_improper flag is set and the classical forcefield cannot match the topology, also the atom index tuples of the improper torsions must be provided in the form of:
        top_dict["impropers"] = [[1,2,3,4], [5,6,7,8], ...]

        Returns a parameter dict containing index tuples (corresponding to the atom_idx passed in the atoms list) and np.ndarrays:
        
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

        "{proper/improper}_ns":np.array of shape (#4-body-terms, n_periodicity), the periodicities of the cos terms of torsion. n_periodicity is a hyperparemter of the model and defaults to 6.

        "{proper/improper}_phases":np.array of shape (#4-body-terms, n_periodicity), the phases of the cos terms of torsion. n_periodicity is a hyperparameter of the model and defaults to 6.

        }
        """
        pass


    def __call__(self, top, system_kwargs:Dict=None):
        """
        Return type is dependent of the input type.
        Input must be either an openmm topology, an openmm PDBFile, a path to a PDB file or a path to a gromacs topology file.
        If the input type is an openmm topology or an openmm PDBFile the output is an openmm system.
        If the input is a path to a PDB file, the output is a dictionary containing indices describing the interactions. If the input is a path to a gromacs topology file, the output is a topology file with parameters added. This has to be made evident by the file suffix (.gro or .pdb).

        The input may also be a dictionary with the keys 'atoms', 'bonds' and 'radicals'. In this case, the workflow of this function is:
        bonds -> dgl graph
        atom_properties -> node features
        write lj parameters in the graph
        charge model: atom properties (atom_name,residue,radical_indices, residue_indices) -> charges -> write charges in dgl graph
        model: dgl_graph -> parametrised dgl graph
        parametrised dgl graph -> parameter dict
        """
        input_type = type(top)
        assert input_type in [openmm.app.topology.Topology, openmm.app.PDBFile, str, dict]

        # if input_type == str:
        #     input_file = Path(top)
        #     input_file = input_file.suffix
        #     assert input_file in [".pdb", ".gro"]
        #     input_type = input_file

        g, openff_mol = process_input(top, classical_ff=self.classical_ff)

        g = self.model(g)

        output = process_output(g=g, openff_mol=openff_mol, input_type=input_type, classical_ff=self.classical_ff, topology=top, system_kwargs=system_kwargs, units=self.units)

        return output
    
    def get_unit_strings(self):
        return {k: str(v) for (k,v) in zip(self.units.keys(), self.units.values())}

    def __str__(self):
        return f"grappa.ForceField with units {json.dumps(self.get_unit_strings(), indent=4)}"

    def __repr__(self):
        return self.__str__()
    

    ################################
    # UTILITIES FOR TESTING
    ################################
    @staticmethod
    def get_classical_model(top:openmm.app.Topology, class_ff:openmm.app.ForceField=openmm.app.ForceField("amber99sbildn.xml")):
        """
        Returns a model that acts like the classical forcefield. This can be used to test and compare the workflow to the classical forcefield.
        This model also overwrites the nonbonded parameters!
        """

        def model(g):
            from grappa.ff_utils.classical_ff.parametrize import parametrize_amber
            g = parametrize_amber(g, top, class_ff, suffix="", charge_suffix="", allow_radicals=True)
            return g
        
        return model
    
    @staticmethod
    def get_zero_model():
        """
        Returns a model that acts like the classical forcefield. This can be used to test and compare the workflow to the classical forcefield.
        This model also overwrites the nonbonded parameters!
        """

        def model(g):
            n_periodicity = 6
            n = g.number_of_nodes("n1")
            one_d_zeros = torch.zeros((n,1))
            two_d_zeros = torch.zeros(g.number_of_nodes("n2"),1)
            three_d_zeros = torch.zeros(g.number_of_nodes("n3"), 1)
            n4_zeros = torch.zeros(g.number_of_nodes("n4"),n_periodicity)
            n5_zeros = torch.zeros(g.number_of_nodes("n4_improper"),n_periodicity)

            g.nodes["n1"].data["q"] = one_d_zeros
            g.nodes["n1"].data["mass"] = one_d_zeros
            g.nodes["n1"].data["sigma"] = one_d_zeros
            g.nodes["n1"].data["epsilon"] = one_d_zeros
            g.nodes["n2"].data["k"] = two_d_zeros
            g.nodes["n2"].data["eq"] = two_d_zeros
            g.nodes["n3"].data["k"] = three_d_zeros
            g.nodes["n3"].data["eq"] = three_d_zeros
            g.nodes["n4"].data["k"] = n4_zeros
            g.nodes["n4_improper"].data["k"] = n5_zeros

            return g
        
        return model
    

    def get_energies(self, top:openmm.app.topology.Topology, positions:np.ndarray, class_ff:openmm.app.ForceField=None, delete_force_type=[]):
        """
        MAKE THIS A FREE FUNCTION
        Returns the energy and gradients of the topology for the given positions
        positions: array of shape n_conf * n_atom * 3, in grappa distance units
        """
        from openmm.app import Simulation
        from openmm import LangevinIntegrator
        from openmm import unit
        from . import units as grappa_units
        
        system_kwargs = {"nonbondedMethod":openmm.app.NoCutoff, "removeCMMotion":False}

        if class_ff is None:
            sys = self(top, system_kwargs=system_kwargs)
        else:
            sys = class_ff.createSystem(top, **system_kwargs)

        if len(delete_force_type) > 0:
            i = 0
            while(i < sys.getNumForces()):
                for force in sys.getForces():
                    if any([d.lower() in force.__class__.__name__.lower() for d in delete_force_type]):
                        print("Removing force", force.__class__.__name__)
                        sys.removeForce(i)
                    else:
                        i += 1

        integrator = LangevinIntegrator(0*unit.kelvin, 1/unit.picosecond, 0.001*unit.picoseconds)
        simulation = Simulation(top, sys, integrator)

        assert len(positions.shape) == 3
        assert positions.shape[1] == top.getNumAtoms()
        assert positions.shape[2] == 3


        ff_forces = []
        ff_energies = []
        for pos in positions:
            simulation.context.setPositions(unit.Quantity(pos, grappa_units.DISTANCE_UNIT).value_in_unit(unit.nanometer)) # go to nanometer
            state = simulation.context.getState(
                    getEnergy=True,
                    getForces=True,
                )
            e = state.getPotentialEnergy()
            e = e.value_in_unit(grappa_units.ENERGY_UNIT)
            f = state.getForces(True)
            f = f.value_in_unit(grappa_units.FORCE_UNIT)
            f = np.array(f)
            ff_energies.append(e)
            ff_forces.append(f)

        ff_forces = np.array(ff_forces)
        ff_energies = np.array(ff_energies)

        return ff_energies, -ff_forces