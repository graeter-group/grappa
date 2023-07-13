import openmm
from typing import List, Tuple, Dict, Union, Set, Callable

from openmm.unit import Quantity, radians

import rdkit.Chem.rdchem

import torch
import numpy as np

from .. import units as grappa_units

from .create_graph import utils, tuple_indices, read_heterogeneous_graph

from .create_graph.find_radical import add_radical_residues, get_radicals

import copy

from ..units import RESIDUES

from grappa.constants import MAX_ELEMENT

class sysWriter:
    """
    Class for initializing a dgl graph along with an openmm system.
    The dgl graph is initialized from the interactions in the openmm system.
    Then a parametrize(model) function can be called to parametrize the graph.
    Afterwards, the parameters from the graph are either written in the system or returned as a dictionary.
    This way we can:
        - Handle impropers correctly (otherwise the order of the non-central atoms is unclear)
        - Keep track of the interaction-indices in the system, making it more efficient to write the parameters back in there.

    The workflow is the following:
    __init__: create system from topology and forcefield (infer radicals)
    init_graph: use the system to initialize the graph with interaction indices.

    """
    def __init__(self, top:openmm.app.Topology, classical_ff:openmm.app.ForceField, allow_radicals:bool=True, radical_indices:List[int]=None) -> None:
        """
        If allow radicals is true and no radical indices are provided, uses the classical forcefield to determine the radical indices. For this the force field must fail to match the residues containing radicals. If the radical indices are provided, assume that these are all radicals.
        Only one radical per residue is supported.
        """
        # ===============================
        # CLASS MEMBERS
        self.sys = None
        self.graph = None
        self.radical_indices = None
        self.interaction_indices = None # dictionary that maps from tuple to index. for torsions, the tuple is (atom1, atom2, atom3, atom4, periodicity)
        self.top = None

        self.top_idxs = None # only parametrize these atoms. is zero-based and corresponds to the order in top.atoms() NOTE: write function that returns sub-topology

        self.use_impropers = True

        self.proper_periodicity = 6
        self.improper_periodicity = 3

        self.max_element = MAX_ELEMENT

        self.units_for_dict = None # has to be set if we want to write the parameters as dictionary instead of in the system

        self.charge_model = None # if None, uses the charges from the forcefield. Otherwise, this should be a callable that takes a topology and returns a list of charges in the order of the topology.

        self.charge_dict = {}

        self.classical_ff = classical_ff



        # ===============================
        # initialisation:

        self.top = top
        self.radical_indices = radical_indices
        self.allow_radicals = allow_radicals
        
        if self.radical_indices is None:
            if self.allow_radicals:
                self.radical_indices, _, _ = get_radicals(topology=self.top, forcefield=classical_ff)
            else:
                self.radical_indices = []
        
        else:
            if len(self.radical_indices) > 0 and not self.allow_radicals:
                raise ValueError("Radicals are not allowed, but radical indices were provided.")
            

        if len(self.radical_indices) > 0:
            # enable the classical ff to match the residues.
            self.classical_ff = copy.deepcopy(classical_ff) # nee a deep copy, otherwise this would change the ff outside the class
            self.classical_ff = add_radical_residues(forcefield=classical_ff, topology=self.top)


        self.sys = self.classical_ff.createSystem(top)
    

    def charges_from_model(self)->List[float]:
        if self.charge_model is None:
            return None
        
        if len(self.radical_indices) > 0:
            charges = self.charge_model(self.top, radical_indices=self.radical_indices)
        else:
            charges = self.charge_model(self.top)
        
        return charges



    def init_graph(self, with_parameters:bool=False)->None:
        """
        Initializes the graph from the system, storing the system-indices of the interactions in dictionaries that map from tuple to index. (this could be done more efficiently with ordered lists)
        The actual graph is initialized from converting the system to an rdkit molecule for obtaining ring-membership features. The interaction indices are read from the system only.
        We assume that the order of the atoms in the topology matches the order of the atoms in the system.
        If with_parameters is true, the classical parameters are read from the system and stored in the graph as {parameter_name}_ref.
        TODO: flag whether we want to write a system later or not. if not the dictionary is unnecessary.
        """


        def is_improper_(rd_mol:rdkit.Chem.rdchem.Mol, idxs:Tuple[int,int,int,int], central_atom_idx:int=2)->bool:
            """
            Helper function to check whether the given tuple of indices describes an improper torsion.
            Checks whether the given tuple of indices describes an improper torsion.
            We can assume that the tuples describe either a proper or improper torsion.
            We also assume that the idxs correspind to the indices of the rdkit molecule.
            """
            # check whether the central atom is the connected to all other atoms in the rdkit molecule.

            central_atom = rd_mol.GetAtomWithIdx(idxs[central_atom_idx])

            # get the neighbors of the central atom
            neighbors = set(central_atom.GetNeighbors())

            # for each atom in the torsion, check if it's a neighbor of the central atom
            for i, atom_idx in enumerate(idxs):
                if i != central_atom_idx:  # skip the central atom itself
                    atom = rd_mol.GetAtomWithIdx(atom_idx)
                    if atom not in neighbors:
                        # if one of the atoms is not connected to it, return False
                        return False

            # if all atoms are connected to the central atom, this is an improper torsion
            return True


            
        # create an rdkit molecule
        rd_mol = utils.openmm2rdkit_graph(openmm_top=self.top)
        
        self.interaction_indices = {"n2":{}, "n3":{}, "n4":{}, "n4_improper":{}}
        
        # to differentiate between improper and proper torsions:
        # this is a set of sets of indices describing a proper torsion
        torsions = tuple_indices.torsion_indices(mol=rd_mol, only_torsion_sets=True)

        bond_idxs = None
        angle_idxs = None
        proper_idxs = None
        improper_idxs = None

        charges = None

        # set the interaction indices:


        # charges:
        if not self.charge_model is None and with_parameters:
            charges = self.charges_from_model()


        for force in self.sys.getForces():
            # ========= ERRORS =========
            if isinstance(force, openmm.CustomTorsionForce):
                raise NotImplementedError("Functional form of CustomTorsionForce is not determined, therefore we cannot learn parameters for this type of force in general.")
            elif isinstance(force, openmm.CustomAngleForce):
                raise NotImplementedError("Functional form of CustomAngleForce is not determined, therefore we cannot learn parameters for this type of force in general.")
            elif isinstance(force, openmm.CustomBondForce):
                raise NotImplementedError("Functional form of CustomBondForce is not determined, therefore we cannot learn parameters for this type of force in general.")
            



            # ========= BONDS =========
            # assume that each bond only occurs once in the system
            elif isinstance(force, openmm.HarmonicBondForce):

                n_bonds = force.getNumBonds()

                # create space for the bond indices and parameters
                bond_idxs = np.zeros((n_bonds, 2), dtype=np.int32)
                if with_parameters:
                    bond_ks = np.zeros((n_bonds, 1), dtype=np.float32)
                    bond_eqs = np.zeros((n_bonds, 1), dtype=np.float32)


                for i in range(force.getNumBonds()):
                    p1, p2, eq, k = force.getBondParameters(i)

                    k = k.value_in_unit(grappa_units.FORCE_CONSTANT_UNIT)
                    eq = eq.value_in_unit(grappa_units.DISTANCE_UNIT)
                        
                    self.interaction_indices["n2"][(p1,p2)] = i
                    bond_idxs[i][0], bond_idxs[i][1] = p1, p2

                    if with_parameters:
                        bond_ks[i] = k
                        bond_eqs[i] = eq




            # ========= ANGLES =========
            elif isinstance(force, openmm.HarmonicAngleForce):
                
                n_angles = force.getNumAngles()
            

                angle_idxs = np.zeros((n_angles, 3), dtype=np.int32)
                if with_parameters:
                    angle_ks = np.zeros((n_angles, 1), dtype=np.float32)
                    angle_eqs = np.zeros((n_angles, 1), dtype=np.float32)
                

                for i in range(force.getNumAngles()):
                    p1, p2, p3, eq, k = force.getAngleParameters(i)

                    k = k.value_in_unit(grappa_units.ANGLE_FORCE_CONSTANT_UNIT)
                    eq = eq.value_in_unit(grappa_units.ANGLE_UNIT)


                    self.interaction_indices["n3"][(p1,p2,p3)] = i
                    angle_idxs[i] = [p1, p2, p3]

                    if with_parameters:
                        angle_ks[i] = k
                        angle_eqs[i] = eq




            # ========= PROPER AND IMPROPER TORSIONS =========
            elif isinstance(force, openmm.PeriodicTorsionForce):

                proper_idxs = []
                improper_idxs = []

                n_torsions = force.getNumTorsions()

                if with_parameters:

                    k_proper = np.zeros((n_torsions, self.proper_periodicity), dtype=np.float32)
                    k_improper = np.zeros((n_torsions, self.improper_periodicity), dtype=np.float32)


                for i in range(force.getNumTorsions()):
                    atom1, atom2, atom3, atom4, periodicity, phase, k = force.getTorsionParameters(i)

                    k = k.value_in_unit(grappa_units.TORSION_FORCE_CONSTANT_UNIT)
                    phase = phase.value_in_unit(grappa_units.TORSION_PHASE_UNIT)

                    # improper or proper?
                    idxs = (atom1, atom2, atom3, atom4)
                    is_improper = is_improper_(rd_mol=rd_mol, idxs=idxs, central_atom_idx=2)


                    if not self.use_impropers and is_improper:
                        continue
                   
                    if not is_improper:
                        if periodicity > self.proper_periodicity:
                            raise ValueError(f"Periodicity of {periodicity} is higher than the maximum of {self.proper_periodicity} for proper torsion {atom1,atom2,atom3,atom4}.")
                        
                        self.interaction_indices["n4"][(atom1, atom2, atom3, atom4, periodicity)] = i
                        if not (atom1, atom2, atom3, atom4) in proper_idxs:
                            proper_idxs.append((atom1, atom2, atom3, atom4))

                            if with_parameters:
                                # we just appended to tuple to the indices, so the length of the proper indices -1 is the position in the graph
                                graph_idx_proper = len(proper_idxs)-1
                            
                        else:
                            if with_parameters:
                                # find the index of the proper in the list of propers
                                graph_idx_proper = proper_idxs.index((atom1, atom2, atom3, atom4))
                    
                        if with_parameters:
                            k_proper[graph_idx_proper, periodicity-1] = k if phase == 0 else -k



                    else:
                        if periodicity > self.improper_periodicity:
                            raise ValueError(f"Periodicity of {periodicity} is higher than the maximum of {self.improper_periodicity} for improper torsion {atom1,atom2,atom3,atom4}.")

                        self.interaction_indices["n4_improper"][(atom1, atom2, atom3, atom4, periodicity)] = i
                        if not (atom1, atom2, atom3, atom4) in improper_idxs:
                            improper_idxs.append((atom1, atom2, atom3, atom4))

                            if with_parameters:
                                # we just appended to tuple to the indices, so the length of the improper indices -1 is the position in the graph
                                graph_idx_improper = len(improper_idxs)-1
                                
                            
                        else:
                            if with_parameters:
                                # find the index of the improper in the list of impropers
                                graph_idx_improper = improper_idxs.index((atom1, atom2, atom3, atom4))
                        
                        if with_parameters:
                            k_improper[graph_idx_improper, periodicity-1] = k if phase == 0 else -k


                # make the tensors smaller since all torsion are both proper and improper
                if len(proper_idxs) < n_torsions:
                    if with_parameters:
                        k_proper = k_proper[:len(proper_idxs)]
                if len(improper_idxs) < n_torsions:
                    if with_parameters:
                        k_improper = k_improper[:len(improper_idxs)]



            # ========= NONBONDED =========
            elif isinstance(force, openmm.NonbondedForce):
                if not with_parameters:
                    continue

                charges = np.zeros(force.getNumParticles(), dtype=np.float32)

                assert force.getNumParticles() == self.sys.getNumParticles()
                for i in range(force.getNumParticles()):
                    charge, sigma, epsilon = force.getParticleParameters(i)
                    q = charge.value_in_unit(grappa_units.CHARGE_UNIT)
                    charges[i] = q

        # use the rd_mol to obtain a homogeneous graph
        self.graph = utils.dgl_from_mol(mol=rd_mol, max_element=self.max_element)

        # use the interaction idxs to obtain a heterogeneous (final) graph:
        self.graph = read_heterogeneous_graph.from_homogeneous_and_idxs(g=self.graph, bond_idxs=bond_idxs, angle_idxs=angle_idxs, proper_idxs=proper_idxs, improper_idxs=improper_idxs, use_impropers=self.use_impropers)

        TERMS = self.graph.ntypes

        # add the parameters:
        if with_parameters:
            if "n2" in TERMS:
                self.graph.nodes["n2"].data["k_ref"] = torch.tensor(bond_ks, dtype=torch.float32)
                self.graph.nodes["n2"].data["eq_ref"] = torch.tensor(bond_eqs, dtype=torch.float32)
            if "n3" in TERMS:
                self.graph.nodes["n3"].data["k_ref"] = torch.tensor(angle_ks, dtype=torch.float32)
                self.graph.nodes["n3"].data["eq_ref"] = torch.tensor(angle_eqs, dtype=torch.float32)
            if "n4" in TERMS:
                self.graph.nodes["n4"].data["k_ref"] = torch.tensor(k_proper, dtype=torch.float32)

            if self.use_impropers and "n4_improper" in TERMS:
                self.graph.nodes["n4_improper"].data["k_ref"] = torch.tensor(k_improper, dtype=torch.float32)

            self.graph.nodes["n1"].data["q_ref"] = torch.tensor(charges, dtype=torch.float32)

        # add residue one-hot and is_radicals:
        self.write_one_hots()


    def write_one_hots(self)->None:
        # write the radicals in the graph:
        is_radical = torch.zeros(self.graph.num_nodes("n1"))
        is_radical[np.array(self.radical_indices)] = 1
        self.graph.nodes["n1"].data["is_radical"] = is_radical.float().unsqueeze(dim=-1) # add a dimension to be able to concatenate with the other features

        
        self.graph.nodes["n1"].data["residue"] = torch.zeros(self.graph.num_nodes("n1"), len(RESIDUES), dtype=torch.float32)

        for idx, a in enumerate(self.top.atoms()):
            resname = a.residue.name
            if resname in RESIDUES:
                res_index = RESIDUES.index(resname) # is unique
                self.graph.nodes["n1"].data["residue"][idx] = torch.nn.functional.one_hot(torch.tensor((res_index)).long(), num_classes=len(RESIDUES)).float()
            else:
                raise ValueError(f"Residue {resname} not in {RESIDUES}")


    def forward_pass(self, model:Callable, device="cpu")->None:
        """
        Writes the parameters predicted by the model into the graph. The model must already be on the given device.
        """
        self.graph = self.graph.to(device)
        self.graph = model(self.graph)
        self.graph = self.graph.cpu()


    def update_system(self)->None:
        """
        Updates the system parameters from the graph.
        """

        # ========= SET THE CHARGES =========
        # change the charges to the ones in the charge_dict
        if (not self.charge_model is None) and self.charge_dict is None:
            self.calc_charge_dict()
            for force in self.sys.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    for i in range(force.getNumParticles()):
                        _, sig, eps = force.getParticleParameters(i)
                        q = self.charge_dict[i]
                        force.setParticleParameters(i, q, sigma, epsilon)
                    for i in range(force.getNumExceptions()):
                        (
                            idx0,
                            idx1,
                            _,
                            sigma,
                            epsilon,
                        ) = force.getExceptionParameters(i)
                        force.setExceptionParameters(
                            i, idx0, idx1, q, sigma, epsilon
                        )


        self.graph = self.graph.to("cpu")

        if "n4" in self.graph.ntypes:
            assert self.graph.nodes["n4"].data["k"].shape[1] == self.proper_periodicity
        if self.use_impropers and "n4_improper" in self.graph.ntypes:
            assert self.graph.nodes["n4_improper"].data["k"].shape[1] == self.improper_periodicity

        for force in self.sys.getForces():
            # ========= ERRORS =========
            if isinstance(force, openmm.CustomTorsionForce):
                raise NotImplementedError("Functional form of CustomTorsionForce is not determined, therefore we cannot learn parameters for this type of force in general.")
            elif isinstance(force, openmm.CustomAngleForce):
                raise NotImplementedError("Functional form of CustomAngleForce is not determined, therefore we cannot learn parameters for this type of force in general.")
            elif isinstance(force, openmm.CustomBondForce):
                raise NotImplementedError("Functional form of CustomBondForce is not determined, therefore we cannot learn parameters for this type of force in general.")

            # ========= BONDS =========
            # assume that each bond only occurs once in the system
            elif isinstance(force, openmm.HarmonicBondForce):
                if force.getNumBonds() == 0:
                    continue
                if force.getNumBonds() != self.graph.num_nodes("n2"):
                    raise ValueError("Number of bonds in the system does not match the number of bonds in the graph.")
                
                eqs = self.graph.nodes["n2"].data["eq"].detach.numpy()
                ks = self.graph.nodes["n2"].data["k"].detach.numpy()

                # convert to openmm units:
                eqs = Quantity(eqs, unit=grappa_units.DISTANCE_UNIT).value_in_unit_system(grappa_units.OPENMM_BOND_EQ_UNIT)
                ks = Quantity(ks, unit=grappa_units.FORCE_CONSTANT_UNIT).value_in_unit_system(grappa_units.OPENMM_BOND_K_UNIT)

                # loop through the graph parameters and update the force:
                for i_graph, (idx_tuple, eq, k) in enumerate(zip(self.graph.nodes["n2"].data["idxs"], eqs, ks)):
                    eq = self.graph.nodes["n2"].data["eq"][i_graph]
                    k = self.graph.nodes["n2"].data["k"][i_graph]
                    particle1, particle2 = idx_tuple
                    # find the index of the bond in the force:
                    idx = self.interaction_indices["n2"][idx_tuple]
                    force.setBondParameters(idx, particle1=particle1, particle2=particle2, length=eq, k=k)

            # ========= ANGLES =========
            # assume that each angle only occurs once in the system
            elif isinstance(force, openmm.HarmonicAngleForce):

                if force.getNumAngles() == 0:
                    continue
                if force.getNumAngles() != self.graph.num_nodes("n3"):
                    raise ValueError("Number of angles in the system does not match the number of angles in the graph.")

                eqs = self.graph.nodes["n3"].data["eq"].detach.numpy()
                ks = self.graph.nodes["n3"].data["k"].detach.numpy()

                # convert to openmm units:
                eqs = Quantity(eqs, unit=grappa_units.ANGLE_UNIT).value_in_unit_system(grappa_units.OPENMM_ANGLE_EQ_UNIT)
                ks = Quantity(ks, unit=grappa_units.ANGLE_FORCE_CONSTANT_UNIT).value_in_unit_system(grappa_units.OPENMM_ANGLE_K_UNIT)

                # loop through the graph parameters and update the force:
                for i_graph, (idx_tuple, eq, k) in enumerate(zip(self.graph.nodes["n3"].data["idxs"], eqs, ks)):
                    particle1, particle2, particle3 = idx_tuple
                    # find the index of the bond in the force:
                    idx = self.interaction_indices["n3"][idx_tuple]
                    force.setAngleParameters(idx, particle1=particle1, particle2=particle2, particle3=particle3, angle=eq, k=k)

            # ========= PROPER AND IMPROPER TORSIONS =========
            elif isinstance(force, openmm.PeriodicTorsionForce):
                if force.getNumTorsions() == 0:
                    continue

                levels = ["n4", "n4_improper"] if self.use_impropers else ["n4"]

                for level in levels:

                    if level not in self.graph.ntypes:
                        continue
                    
                    if level == "n4":
                        assert self.graph.nodes["n4"].data["k"].shape[1] == self.proper_periodicity
                    elif level == "n4_improper":
                        assert self.graph.nodes["n4_improper"].data["k"].shape[1] == self.improper_periodicity

                    ns, phases, k_vec = sysWriter.get_periodicity_phase_k(ks=self.graph.nodes[level].data["k"])
                    idxs = self.graph.nodes[level].data["idxs"]
                    
                    # convert to openmm units
                    phases = Quantity(phases, unit=grappa_units.ANGLE_UNIT).value_in_unit(grappa_units.OPENMM_TORSION_PHASE_UNIT)
                    k_vec = Quantity(k_vec, unit=grappa_units.TORSION_FORCE_CONSTANT_UNIT).value_in_unit(grappa_units.OPENMM_TORSION_K_UNIT)

                    for i_graph, idx_tuple in enumerate(idxs):

                        for periodicity, phase, k in zip(ns[i_graph], phases[i_graph], k_vec[i_graph]):
                            k = Quantity(k, unit=grappa_units.ENERGY_UNIT)
                            phase = Quantity(phase, unit=grappa_units.ANGLE_UNIT)
                            key = idx_tuple + (periodicity,)
                            particle1, particle2, particle3, particle4 = idx_tuple

                            # if this torsion was in the original system, update it
                            if key in self.interaction_indices:
                                idx = self.interaction_indices[level][key]
                                force.setTorsionParameters(idx, particle1=particle1, particle2=particle2, particle3=particle3, particle4=particle4, periodicity=periodicity, phase=eq, k=k)
                            # otherwise, add it
                            else:
                                force.addTorsion(particle1=particle1, particle2=particle2, particle3=particle3, particle4=particle4, periodicity=periodicity, phase=phase, k=k)

    @staticmethod
    def get_periodicity_phase_k(ks:Union[torch.Tensor, np.ndarray])->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(ks, torch.Tensor):
            ks = ks.detach().cpu().numpy()

        ns = np.arange(1,ks.shape[1]+1) # one set of periodicities
        ns = np.tile(ns, (ks.shape[0], 1)) # repeat this for all torsions.

        pi_in_units = Quantity(np.pi, unit=radians).value_in_unit(grappa_units.ANGLE_UNIT)

        phases = np.where(ks>0, np.zeros_like(ks), pi_in_units*np.ones_like(ks)) # set the phases to pi for negative ks.
        ks = np.abs(ks) # take the absolute value of the ks.

        return ns, phases, ks
    

    def write_confs(self, xyz:Union[np.ndarray, torch.Tensor], qm_energies:Union[np.ndarray, torch.Tensor]=None, qm_gradients:Union[np.ndarray, torch.Tensor]=None)->None:
        """
        Write the xyz coordinates to the graph. Calculate total and nonbonded energy/gradients of the classical forcefield and write them to the graph.
        xyz: Array of shape (N_conf x N_atoms x 3) containing the atom positions in grappa.units.
        qm_energies: Array of shape (N_conf) containing the energies in grappa.units.
        qm_gradients: Array of shape (N_conf x N_atoms x 3) containing the gradients in grappa.units.
        """
        # only dummies for simulation intialization
        TEMPERATURE = 350 * openmm.unit.kelvin
        STEP_SIZE = 1.0 * openmm.unit.femtosecond
        COLLISION_RATE = 1.0 / openmm.unit.picosecond


        # calculate the energies and gradients:
        # create a copy of the system
        system = copy.deepcopy(self.sys)


        # initialize an integrator to be able to initialize a simulation to calculate energies:
        integrator = openmm.LangevinIntegrator(
            TEMPERATURE, COLLISION_RATE, STEP_SIZE
        )
            # create simulation
        simulation = openmm.app.Simulation(
            topology=self.top, system=system, integrator=integrator
        )


        # ========= SET THE CHARGES =========

        if "q_ref" in self.graph.nodes["n1"].data:
            for force in system.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    for i in range(force.getNumParticles()):
                        _, sig, eps = force.getParticleParameters(i)
                        q = self.graph.nodes["n1"].data["q_ref"][i]
                        q = Quantity(q, unit=grappa_units.CHARGE_UNIT).value_in_unit(grappa_units.OPENMM_CHARGE_UNIT)

                        force.setParticleParameters(i, q, sig, eps)
                    for i in range(force.getNumExceptions()):
                        (
                            idx0,
                            idx1,
                            q,
                            sigma,
                            epsilon,
                        ) = force.getExceptionParameters(i)
                        force.setExceptionParameters(
                            i, idx0, idx1, q, sigma, epsilon
                        )

                    force.updateParametersInContext(simulation.context)


        # ========= TOTAL ENERGIES ==========
    
        total_energies, total_gradients = sysWriter.get_energies_(simulation, np.array(xyz))
        
        
        # ========= SET BONDED PARAMETERS TO ZERO =========
        # turn off angle:
        for force in system.getForces():
            if isinstance(force, openmm.HarmonicAngleForce):
                for idx in range(force.getNumAngles()):
                    id1, id2, id3, angle, k = force.getAngleParameters(idx)
                    force.setAngleParameters(idx, id1, id2, id3, angle, 0.0)

                force.updateParametersInContext(simulation.context)
        
            if isinstance(force, openmm.HarmonicBondForce):
                for idx in range(force.getNumBonds()):
                    id1, id2, length, k = force.getBondParameters(idx)
                    force.setBondParameters(
                        idx,
                        id1,
                        id2,
                        length,
                        0.0,
                    )

                force.updateParametersInContext(simulation.context)
        
            # also contains improper torsions
            if isinstance(force, openmm.PeriodicTorsionForce):
                for idx in range(force.getNumTorsions()):
                    (
                        id1,
                        id2,
                        id3,
                        id4,
                        periodicity,
                        phase,
                        k,
                    ) = force.getTorsionParameters(idx)

                    force.setTorsionParameters(
                        idx,
                        id1,
                        id2,
                        id3,
                        id4,
                        periodicity,
                        phase,
                        0.0,
                    )
                force.updateParametersInContext(simulation.context)


        # ========= NONBONDED ENERGIES =========
        nonbonded_energies, nonbonded_gradients = sysWriter.get_energies_(simulation, np.array(xyz))


        # write the coordinates to the graph
        self.graph.nodes["n1"].data["xyz"] = torch.tensor(xyz, dtype=torch.float32).transpose(0,1)

        # write the energies to the graph
        self.graph.nodes["g"].data["u_total_ref"] = torch.tensor(total_energies, dtype=torch.float32).unsqueeze(dim=0)
        self.graph.nodes["g"].data["u_nonbonded_ref"] = torch.tensor(nonbonded_energies, dtype=torch.float32).unsqueeze(dim=0)

        # write the gradients to the graph
        self.graph.nodes["n1"].data["grad_total_ref"] = torch.tensor(total_gradients, dtype=torch.float32).transpose(0,1)
        self.graph.nodes["n1"].data["grad_nonbonded_ref"] = torch.tensor(nonbonded_gradients, dtype=torch.float32).transpose(0,1)
        

        # ========= XYZ AND QM DATA =========

        self.graph.nodes["n1"].data["xyz"] = torch.tensor(xyz, dtype=torch.float32).transpose(0,1)

        if not qm_energies is None:
            self.graph.nodes["g"].data["u_qm"] = torch.tensor(qm_energies, dtype=torch.float32).unsqueeze(dim=0)
            self.graph.nodes["g"].data["u_ref"] = torch.tensor(qm_energies - nonbonded_energies, dtype=torch.float32).unsqueeze(dim=0)

        if not qm_gradients is None:
            self.graph.nodes["n1"].data["grad_qm"] = torch.tensor(qm_gradients, dtype=torch.float32).transpose(0,1)
            self.graph.nodes["n1"].data["grad_ref"] = torch.tensor(qm_gradients - nonbonded_gradients, dtype=torch.float32).transpose(0,1)


        # ========= USE_K =========
        # preparation for training
        terms = ["n4", "n4_improper"] if self.use_impropers else ["n4"]
        for t in terms:
            if t in self.graph.ntypes:
                self.graph.nodes[t].data["use_k"] = torch.where(torch.abs(self.graph.nodes[t].data[f"k_ref"]) > 0, 1., 0.)



    @staticmethod
    def get_energies_(simulation, xs)->Tuple[np.ndarray, np.ndarray]:
        """
        helper function
        returns the total energies and gradients of a simulation
        shape of xyz and gradients:
            (n_samples, n_atoms, 3)
        """

        xs = Quantity(
                xs,
                grappa_units.DISTANCE_UNIT,
            ).value_in_unit(grappa_units.OPENMM_LENGTH_UNIT)
        
        # loop through the snapshots
        energies = []
        derivatives = []

        for x in xs:
            simulation.context.setPositions(x)

            state = simulation.context.getState(
                getEnergy=True,
                getParameters=True,
                getForces=True,
            )

            energy = state.getPotentialEnergy().value_in_unit(
                grappa_units.ENERGY_UNIT,
            )


            forces = state.getForces(asNumpy=True).value_in_unit(
                grappa_units.FORCE_UNIT,
            )

            energies.append(energy)
            derivatives.append(-forces)

        energies = np.array(energies, dtype=np.float32)
        derivatives = np.array(derivatives, dtype=np.float32)

        return energies, derivatives