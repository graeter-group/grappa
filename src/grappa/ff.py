"""
Class wrapping a model and providing methods for translating various input types to dgl graphs which can be processed by the model and translate back to various output types.
"""


import torch
import dgl
from typing import Union, List, Tuple, Dict, Callable
from .ff_utils.classical_ff.collagen_utility import get_collagen_forcefield
from .models.deploy import model_from_path, model_from_tag
import openmm.app.topology
import openmm.app
from pathlib import Path
import numpy as np
import json

from .ff_utils.SysWriter import SysWriter
from .ff_utils.SysWriter import TopologyDict, ParamDict

from .ff_utils.charge_models.charge_models import model_from_dict

from openmm import unit

class ForceField:
    def __init__(self, model:Callable=None, model_path:Union[str, Path]=None, classical_ff:openmm.app.ForceField=openmm.app.ForceField("amber99sbildn.xml"), charge_model:Union[str,Callable]=None, allow_radicals:bool=False, device:str="cpu") -> None:
        """
        Class wrapping a model and providing methods for translating various input types to dgl graphs which can be processed by the model and translate back to various output types.
        model_path: a path to a folder with a single .pt file containing a model-state_dict and a config.yaml file containing model hyperparameters for construction.
        model: a callable taking and returning a dgl graph.
        classical_ff: an openmm forcefield object used for nonbonded parameters and system initialization.
        units: a dictionary containing the openmm.unit used for the output dictionary. The charge unit is always elementary charges.

            The keys and default values are:
            {
                "distance": unit.nanometer,
                "angle": unit.radian,
                "energy": unit.kilojoule_per_mole,
            }
            Note that the derived parameters then are:
                BOND_EQ_UNIT = DISTANCE_UNIT
                ANGLE_EQ_UNIT = ANGLE_UNIT
                TORSION_K_UNIT = ENERGY_UNIT
                TORSION_PHASE_UNIT = ANGLE_UNIT
                BOND_K_UNIT = ENERGY_UNIT / (DISTANCE_UNIT**2)
                ANGLE_K_UNIT = ENERGY_UNIT / (ANGLE_UNIT**2)

        """

        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device


        if model is not None and model_path is not None:
            raise ValueError("Either model or model_path must be given, not both.")
        if model is None and model_path is None:
            raise ValueError("Either model or model_path must be given.")

        if model is not None:
            self.model = model
        else:
            self.model = model_from_path(model_path, device=self.device)
            self.model.eval()

        
        self.classical_ff = classical_ff # used to create the dgl.graph, i.e. to obtain indices of impropers, propers, angles and bonds.

        self.units = {
            "distance": unit.nanometer,
            "angle": unit.radian,
            "energy": unit.kilojoule_per_mole,
        }

        self.use_improper = True # if False, does not use impropers, which allows for the prediction of parameters without the need of a classical forcefield. The reason is that improper ordering and which ones to use at all are not unique.
        assert self.use_improper, "Currently, the use_improper flag must be True."

        self.set_charge_model(charge_model)

        self.allow_radicals = allow_radicals


    @classmethod
    def from_tag(cls, tag:str, device:str="cpu")->"ForceField":
        """
        Initializes the ForceField from a tag. Available tags:

        example - An example model, not fine-tuned for good performance. Builds upon an extension of amber99sbildn for DOP and HYP. Allows radicals, uses the 'heavy' charge model. For ParamDict, uses degrees instead of radians for angles and torsions.

        """

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = device

        if tag == "example":
            model_tag = "example"
            model = model_from_tag(model_tag, device=device)
            classical_ff = get_collagen_forcefield()
            charge_model = "heavy"
            allow_radicals = True

            self = cls(model=model, classical_ff=classical_ff, charge_model=charge_model, allow_radicals=allow_radicals, device=device)

            self.units["angle"] = unit.degree
            
            return self

        else:
            raise ValueError(f"Unknown tag {tag}.")


    def set_charge_model(self, charge_model:Union[Callable, str, None])->None:
        """
        Sets the charge model to the given callable. May also be a tag.
        The callable must take a openmm.app.Topology and return a list of charges in grappa units.
        Possible tags are:
        'bmk' - 'own charges using a scheme to obtain tbulated charges from a QM-calculated electron density
        'heavy' - 'the charges from aber99sbildn where in case of a radical of type 'H-missing' the heavy atom at which the hydrogen is missing receives the partial charge from the hydrogen.
        'avg': 'the charges from aber99sbildn where in case of a radical of type 'H-missing' all atoms receive an equal share of the partial charge from the hydrogen.
        """
        if isinstance(charge_model, str):
            self.charge_model = model_from_dict(tag=charge_model)
        else:
            self.charge_model = charge_model


    def createSystem(self, topology:openmm.app.Topology, **system_kwargs)->openmm.System:
        """
        Returns:
            An openmm.System with the given topology and the parameters predicted by the model.

        Accepts:
            An openmm.app.Topology describing the system to be parametrised.

        Parametrises the topology using the internal model and returns an openmm.System describing the topology with the predicted parameters.
        """
        writer = SysWriter(top=topology, allow_radicals=self.allow_radicals, classical_ff=self.classical_ff, **system_kwargs)
        writer.set_charge_model(self.charge_model)
        writer.init_graph(with_parameters=False)
        writer.forward_pass(model=self.model, device=self.device)
        writer.update_system()
    
        return writer.sys

    
    def system_from_topology_dict(self, topology:TopologyDict, **system_kwargs)->openmm.System:
        """
        Returns:
            An openmm.System with the given topology and the parameters predicted by the model.

        Accepts:
            A :any:`~grappa.constants.TopologyDict`, i.e. a dict with the keys 'atoms', 'bonds' and, optionally, 'radicals'.

            Sigmas and epsilons must not be given, all are calculated by the classical forcefield if their entry is None for some atom.

        Parametrises the topology using the internal model and returns an openmm.System describing the topology with the predicted parameters.
        """
        
        writer = SysWriter.from_dict(topology=topology, ordered_by_res=True, allow_radicals=self.allow_radicals, classical_ff=self.classical_ff, **system_kwargs)
        writer.set_charge_model(self.charge_model)
        writer.init_graph(with_parameters=False)
        writer.forward_pass(model=self.model, device=self.device)
        writer.update_system()
    
        return writer.sys
        

    def params_from_topology_dict(self, topology:TopologyDict)->ParamDict:
        """
        Returns:
            A dictionary containing the parameters predicted by the model for the given topology.

        Accepts:
            A :any:`~grappa.constants.TopologyDict`, i.e. a dict with the keys 'atoms', 'bonds' and, optionally, 'radicals'.

            Sigmas and epsilons must not be given, all are calculated by the classical forcefield if their entry is None for some atom.

        Parametrises the topology useing the internal model and returns a parameter dict containing index tuples (corresponding to the atom_idx passed in the atoms list) and np.ndarrays:
        
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
        writer = SysWriter.from_dict(topology=topology, ordered_by_res=True, allow_radicals=self.allow_radicals, classical_ff=self.classical_ff)
        writer.set_charge_model(self.charge_model)
        writer.init_graph(with_parameters=False)
        writer.forward_pass(model=self.model, device=self.device)
        d = writer.get_parameter_dict(units=self.units)
        return d


    def __call__(self, topology:Union[TopologyDict, openmm.app.Topology], **system_kwargs)->Union[ParamDict, openmm.System]:
        """
        Depending on the input type, either calls self.params_from_topology_dict or self.createSystem.
        """
        if isinstance(topology, Dict):
            assert len(list(system_kwargs.keys())) == 0, "system_kwargs must be empty if topology is a TopologyDict"

            return self.params_from_topology_dict(topology=topology)
        
        elif isinstance(topology, openmm.app.Topology):
            return self.createSystem(topology=topology, **system_kwargs)
        
        else:
            raise TypeError(f"Expected openmm.app.Topology or TopologyDict, got {type(topology)}")


    
    def get_unit_strings(self):
        return {k: str(v) for (k,v) in zip(self.units.keys(), self.units.values())}

    def __str__(self):
        out = f"grappa.ForceField with"
        
        out += f"\nunits:\n{json.dumps(self.get_unit_strings(), indent=4)}"

        out += f"\nallow_radicals: {self.allow_radicals}"

        return out+"\n"

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
            writer = SysWriter(top=top, classical_ff=class_ff)
            writer.init_graph(with_parameters=True)

            # rename the ref features:
            for level in writer.graph.ntypes:
                static_feats = [feat for feat in writer.graph.nodes[level].data.keys() if "_ref" == feat[-4:]]
                for feat in static_feats:
                    new_feat = feat[:-4]
                    writer.graph.nodes[level].data[new_feat] = writer.graph.nodes[level].data[feat]

            if "xyz" in g.nodes["n1"].data.keys():
                xyz = g.nodes["n1"].data["xyz"]
                writer.graph.nodes["n1"].data["xyz"] = xyz
                
            return writer.graph
        
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

            if "n4_improper" in g.ntypes:
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
            if "n4_improper" in g.ntypes:
                g.nodes["n4_improper"].data["k"] = n5_zeros


            return g
        
        return model
    

    def get_energies(self, topology:openmm.app.topology.Topology, positions:np.ndarray, class_ff:openmm.app.ForceField=None, delete_force_type=[]):
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
            sys = self.createSystem(topology=topology, **system_kwargs)
        else:
            sys = class_ff.createSystem(topology=topology, **system_kwargs)

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
        simulation = Simulation(topology, sys, integrator)

        assert len(positions.shape) == 3
        assert positions.shape[1] == topology.getNumAtoms()
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