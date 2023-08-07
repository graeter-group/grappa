import openmm
from typing import List, Tuple, Dict, Union, Set, Callable

from openmm.unit import Quantity, radians

import rdkit.Chem.rdchem

from grappa.ff_utils.charge_models.charge_models import model_from_dict

import torch
import numpy as np

from .. import units as grappa_units

from .create_graph import utils, tuple_indices, read_heterogeneous_graph, chemical_features

from grappa.ff_utils.classical_ff.collagen_utility import get_mod_amber99sbildn


from .create_graph.find_radical import add_radical_residues, get_radicals

import copy

from ..units import RESIDUES

from grappa.constants import MAX_ELEMENT, TopologyDict, ParamDict

# todo:
# filter topology for residues that are in grappa.constants.RESIDUES -> subtopology
# write interal->external index things
# make update system dependent on the existence of these index trafos (in this case, new system has to be created)
# should be trivial except for nonbonded. there we have to go through all exceptions and replace the for .. in particles by for .. in param_dict 

class SysWriter:
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
    def __init__(self,
                 top:openmm.app.Topology,
                 classical_ff:openmm.app.ForceField=get_mod_amber99sbildn(),
                 allow_radicals:bool=True,
                 radical_indices:List[int]=None,
                 smiles_flag:bool=False,
                 **system_kwargs) -> None:
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

        self.classical_ff = classical_ff


        self.top_idxs = None # only parametrize these atoms. is zero-based and corresponds to the order in top.atoms() NOTE: write function that returns sub-topology

        self.use_impropers = True

        self.proper_periodicity = 6
        self.improper_periodicity = 3

        self.max_element = MAX_ELEMENT

        self.units_for_dict = None # has to be set if we want to write the parameters as dictionary instead of in the system

        self.charge_model = None # if None, uses the charges from the forcefield. Otherwise, this should be a callable that takes a topology and returns a list of charges in the order of the topology.

        self.epsilons = None # will be used upon parametrization by dict if not None. Only used in a parameter dict, not in createSystem. (due to nonbonded exceptions)
        self.sigma = None

        self.external_to_internal_idx = None # dict that maps from external to internal indices. Internal indices are the zero-based indices in the graph and of the system. if None, assume that the indices are the same.

        self.internal_to_external_idx = None # np.array that maps from internal (array position) to external (array value) indices. Internal indices are the zero-based indices in the graph and of the system. if None, assume that the indices are the same.


        self.use_residues = True # whether to store residues one-hot encoded in the graph

        self.additional_features = None # additional features to be stored in the graph. must be a torch tensor of shape (n_atoms, n_features) with dtype torch.float32

        self.cyclic_impropers = True # whether to store 3 impropers in the graph or only one. if True, stores not only (1,2,3,4) but also (2,4,3,1) and (4,1,3,2). if False, only stores (1,2,3,4).

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
            self.classical_ff = add_radical_residues(forcefield=self.classical_ff, topology=self.top)

        if not self.classical_ff is None:
            self.sys = self.classical_ff.createSystem(top, **system_kwargs)

        if self.classical_ff is None and not smiles_flag:
            raise ValueError("No forcefield provided.")

    @classmethod
    def from_bonds(cls,
                   bonds:List[Tuple[int, int]],
                   residue_indices:List[int],
                   residues:List[str],
                   atom_types:List[str],
                   atomic_numbers:List[int],
                   classical_ff:openmm.app.ForceField=get_mod_amber99sbildn(),
                   ordered_by_res:bool=True,
                   atom_indices:List[int]=None,
                   radicals:List[int]=None,
                   allow_radicals:bool=True,
                   **system_kwargs) -> "SysWriter":


        external_to_internal_idx = None

        if atom_indices is not None:
            external_to_internal_idx = {atom_indices[i]:i for i in range(len(atom_indices))} # i-th entry is the list-position of the atom with index i

            bonds = [(external_to_internal_idx[bond[0]], external_to_internal_idx[bond[1]]) for bond in bonds]

            radical_indices = [external_to_internal_idx[radical] for radical in radicals]

        # get an openmm topology:
        top = utils.bonds_to_openmm(bonds=bonds, residue_indices=residue_indices, residues=residues, atom_types=atom_types, atomic_numbers=atomic_numbers, ordered_by_res=ordered_by_res)

        self = cls(top=top, classical_ff=classical_ff, allow_radicals=allow_radicals, radical_indices=radical_indices, **system_kwargs)

        self.external_to_internal_idx = external_to_internal_idx
        
        self.internal_to_external_idx = np.array(atom_indices, dtype=np.int64)

        return self



    @classmethod
    def from_dict(cls,
                  topology:TopologyDict,
                  ordered_by_res=True,
                  allow_radicals:bool=True,
                  classical_ff:openmm.app.ForceField=get_mod_amber99sbildn(),
                  **system_kwargs) -> "SysWriter":
        """
        Sigmas and epsilons must not be given, all are calculated by the classical forcefield if their entry is None for some atom.
        """

        atoms = topology["atoms"]

        external_idxs = [a_entry[0] for a_entry in atoms]
        atom_types = [a_entry[1] for a_entry in atoms]
        residues = [a_entry[2] for a_entry in atoms]
        atomic_numbers = [a_entry[5] for a_entry in atoms]
        residue_indices = [a_entry[3] for a_entry in atoms]

        # store these as arrays and write them to the parameter dict later:
        try:
            epsilons = [a_entry[4][1] for a_entry in atoms]
            sigmas = [a_entry[4][0] for a_entry in atoms]
        except:
            epsilons = None
            sigmas = None
        

        bonds = topology["bonds"]

        if "radicals" in topology.keys():
            radicals = topology["radicals"]
        else:
            radicals = None


        self = cls.from_bonds(bonds=bonds, residue_indices=residue_indices, residues=residues, atom_types=atom_types, atom_indices=external_idxs, radicals=radicals, allow_radicals=allow_radicals, ordered_by_res=ordered_by_res, atomic_numbers=atomic_numbers, classical_ff=classical_ff, **system_kwargs)

        self.set_lj(epsilons=np.array(epsilons, dtype=np.float32), sigmas=np.array(sigmas, dtype=np.float32))
        
        return self
    


    @classmethod
    def from_smiles(cls, smiles:str, ff="gaff-2.11", **system_kwargs):
        """
        Creates a SysWriter from a smiles string and a small-molecule forcefield. This enables compararision with espaloma.
        """

        # supress warnings from openff imports that are optional:
        from .create_graph.openff_imports import get_openff_imports

        SystemGenerator, Molecule = get_openff_imports()


        mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
        top = mol.to_topology().to_openmm()

        amber_forcefields = ['amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml', 'amber/tip3p_HFE_multivalent.xml']

        forcefield_kwargs = system_kwargs

        system_generator = SystemGenerator(forcefields=amber_forcefields, small_molecule_forcefield=ff, forcefield_kwargs=forcefield_kwargs)

        # initialize an empty object:
        self = cls(top=top, classical_ff=None, allow_radicals=False, smiles_flag=True)


        self.sys = system_generator.create_system(
            topology=mol.to_topology().to_openmm(),
            molecules=mol,
        )

        self.use_residues = False

        mol = mol.to_rdkit()
        self.additional_features = chemical_features.get_chemical_features(mol)

        return self


    def set_charge_model(self, charge_model:Union[Callable, str, None])->None:
        """
        Sets the charge model to the given callable. May also be a tag.
        """
        if isinstance(charge_model, str):
            self.charge_model = model_from_dict(tag=charge_model)
        else:
            self.charge_model = charge_model


    def set_lj(self, epsilons:np.ndarray, sigmas:np.ndarray):
        """
        Stores the given epsilons and sigmas internally. Will be used upon parametrization.
        """
        self.epsilons = epsilons
        self.sigmas = sigmas


    def charges_from_model(self)->List[float]:
        """
        Internal helper.
        """
        if self.charge_model is None:
            return None
        
        if len(self.radical_indices) > 0:
            charges = self.charge_model(self.top, radical_indices=self.radical_indices)
        else:
            charges = self.charge_model(self.top)
        
        return np.array(charges, dtype=np.float32)
    

    def charges_from_ff(self, set_lj:bool=False)->List[float]:
        """
        Internal helper. If set_lj is True, also stores the LJ parameters internally.
        """
        nonbonded_counter = 0
        for force in self.sys.getForces():
            if isinstance(force, openmm.NonbondedForce):
                assert force.getNumParticles() == self.sys.getNumParticles()
                nonbonded_counter += 1

                if set_lj:
                    charges = np.zeros(self.sys.getNumParticles(), dtype=np.float32)
                    self.epsilons = np.zeros(self.sys.getNumParticles(), dtype=np.float32)
                    self.sigmas = np.zeros(self.sys.getNumParticles(), dtype=np.float32)
                    for i in range(force.getNumParticles()):
                        charge, sigma, epsilon = force.getParticleParameters(i)
                        charges[i] = charge.value_in_unit(grappa_units.CHARGE_UNIT)
                        self.epsilons[i] = epsilon.value_in_unit(grappa_units.ENERGY_UNIT)
                        self.sigmas[i] = sigma.value_in_unit(grappa_units.DISTANCE_UNIT)

                else:
                    charges = np.array([force.getParticleParameters(i)[0].value_in_unit(grappa_units.CHARGE_UNIT) for i in range(force.getNumParticles())], dtype=np.float32)

        assert nonbonded_counter == 1, "More or less than one nonbonded force in system."
        return charges


    def init_graph(self, with_parameters:bool=False)->None:
        """
        Initializes the graph from the system, storing the system-indices of the interactions in dictionaries that map from tuple to index. (this could be done more efficiently with ordered lists)
        The actual graph is initialized from converting the system to an rdkit molecule for obtaining ring-membership features. The interaction indices are read from the system only.
        We assume that the order of the atoms in the topology matches the order of the atoms in the system.
        If with_parameters is true, the classical parameters are read from the system and stored in the graph as {parameter_name}_ref.
        TODO: flag whether we want to write a system later or return a param dict. if not the interaction dictionary is unnecessary.
        """


        def is_improper_(rd_mol:rdkit.Chem.rdchem.Mol, idxs:Tuple[int,int,int,int], central_atom_idx:int=2)->bool:
            """
            Helper function to check whether the given tuple of indices describes an improper torsion.
            Checks whether the given tuple of indices describes an improper torsion.
            We can assume that the tuples describe either a proper or improper torsion.
            We also assume that the idxs correspond to the indices of the rdkit molecule.
            """
            # check whether the central atom is the connected to all other atoms in the rdkit molecule.

            central_atom = rd_mol.GetAtomWithIdx(idxs[central_atom_idx])

            # get the neighbors of the central atom
            neighbor_idxs = set([n.GetIdx() for n in central_atom.GetNeighbors()])

            # for each atom in the torsion, check if it's a neighbor of the central atom
            for i, atom_idx in enumerate(idxs):
                if i != central_atom_idx:  # skip the central atom itself
                    if atom_idx not in neighbor_idxs:
                        # if one of the atoms is not connected to it, return False
                        return False

            # if all atoms are connected to the central atom, this is an improper torsion
            return True


        # create an rdkit molecule
        rd_mol = utils.openmm2rdkit_graph(openmm_top=self.top)
        
        self.interaction_indices = {"n2":{}, "n3":{}, "n4":{}, "n4_improper":{}}
        

        bond_idxs = None
        angle_idxs = None
        proper_idxs = None
        improper_idxs = None
        

        # set the interaction indices:


        # charges:
        if not self.charge_model is None and with_parameters:
            self.charges = self.charges_from_model()


        for force in self.sys.getForces():
            # ========= ERRORS =========
            if isinstance(force, openmm.CustomTorsionForce):
                raise NotImplementedError("Functional form of CustomTorsionForce is not determined, therefore we cannot learn parameters for this type of force in general.")
            elif isinstance(force, openmm.CustomAngleForce):
                raise NotImplementedError("Functional form of CustomAngleForce is not determined, therefore we cannot learn parameters for this type of force in general.")
            elif isinstance(force, openmm.CustomBondForce):
                raise NotImplementedError("Functional form of CustomBondForce is not determined, therefore we cannot learn parameters for this type of force in general.")
            
            elif isinstance(force, openmm.NonbondedForce):
                # write the charges in the graph
                charges = torch.zeros(self.sys.getNumParticles(), dtype=torch.float32)
                for i in range(force.getNumParticles()):
                    charge, sigma, epsilon = force.getParticleParameters(i)
                    charges[i] = charge.value_in_unit(grappa_units.CHARGE_UNIT)



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

                n_torsions = force.getNumTorsions()

                proper_idxs = []
                improper_idxs = []


                if with_parameters:

                    k_proper = np.zeros((n_torsions, self.proper_periodicity), dtype=np.float32)
                    k_improper = np.zeros((n_torsions*3, self.improper_periodicity), dtype=np.float32)


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
                            if self.cyclic_impropers:
                                # also append permuted versions of this. we wish to enforce invariance of the enrgy under permutations of the outer atoms. we assume that the central atom is atom3 as usual in amber forcefields: http://archive.ambermd.org/201305/0131.html
                                # we can reduce the permutations necessary from 6 to 3 since the dihedrals are not independent but are antisymmetric under permutation of atoms in positions 1 and 4. therefore we can express the outer-atom-permutation-symmetric energy as an antisymmetric k and 3 different orderings of the outer atoms. 
                                improper_idxs.append((atom2, atom4, atom3, atom1))
                                improper_idxs.append((atom4, atom1, atom3, atom2))

                                # their parameters are kept to zero. a model will not be able to learn to fit these parameters directly, because it is arbitrary which of the 3 permutations is the one that is present in amber.

                            if with_parameters:
                                # we just appended to tuple to the indices, so the length of the improper indices -1 is the position in the graph
                                graph_idx_improper = len(improper_idxs)-1
                                
                            
                        else:
                            if with_parameters:
                                # find the index of the improper in the list of impropers
                                graph_idx_improper = improper_idxs.index((atom1, atom2, atom3, atom4))
                        
                        if with_parameters:
                            k_improper[graph_idx_improper, periodicity-1] = k if phase == 0 else -k


                # make the tensors smaller since (all torsions) are (union of proper and improper)

                if with_parameters:
                    if len(proper_idxs) > 0:
                        k_proper = k_proper[:len(proper_idxs)]
                    if len(improper_idxs) > 0:
                        k_improper = k_improper[:len(improper_idxs)]



        if not proper_idxs is None:
            proper_idxs = np.array(proper_idxs, dtype=np.int32)
        if not improper_idxs is None:
            improper_idxs = np.array(improper_idxs, dtype=np.int32)


        # use the rd_mol to obtain a homogeneous graph
        self.graph = utils.dgl_from_mol(mol=rd_mol, max_element=self.max_element)

        # use the interaction idxs to obtain a heterogeneous (final) graph:
        self.graph = read_heterogeneous_graph.from_homogeneous_and_idxs(g=self.graph, bond_idxs=bond_idxs, angle_idxs=angle_idxs, proper_idxs=proper_idxs, improper_idxs=improper_idxs, use_impropers=self.use_impropers)

        TERMS = self.graph.ntypes

        if "n1" in TERMS:
            self.graph.nodes["n1"].data["q_ref"] = charges.float()

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

        # add residue one-hot and is_radicals:
        self.write_one_hots()


    def write_one_hots(self)->None:
        # write the radicals in the graph:
        is_radical = torch.zeros(self.graph.num_nodes("n1"))
        is_radical[np.array(self.radical_indices)] = 1
        self.graph.nodes["n1"].data["is_radical"] = is_radical.float().unsqueeze(dim=-1) # add a dimension to be able to concatenate with the other features

        
        self.graph.nodes["n1"].data["residue"] = torch.zeros(self.graph.num_nodes("n1"), len(RESIDUES), dtype=torch.float32)

        if self.use_residues:
            for idx, a in enumerate(self.top.atoms()):
                resname = a.residue.name
                if resname in RESIDUES:
                    res_index = RESIDUES.index(resname) # is unique
                    self.graph.nodes["n1"].data["residue"][idx] = torch.nn.functional.one_hot(torch.tensor((res_index)).long(), num_classes=len(RESIDUES)).float()
                else:
                    raise ValueError(f"Residue {resname} not in {RESIDUES}")

        if not self.additional_features is None:
            self.graph.nodes["n1"].data["additional_features"] = self.additional_features


    def forward_pass(self, model:Callable, device="cpu")->None:
        """
        Writes the parameters predicted by the model into the graph. The model must already be on the given device.
        """
        self.graph = self.graph.to(device)
        with torch.no_grad():
            self.graph = model(self.graph)
        self.graph = self.graph.cpu()


    def get_parameter_dict(self, units:Dict=None)->ParamDict:
        """
        Returns a dictionary with the parameters of the molecule.
        Units are given by the unit_dict. By default, the units are openmm units, i.e.
        {
            "distance":openmm.unit.nanometer,
            "angle":openmm.unit.radian,
            "energy":openmm.unit.kilojoule_per_mole,
        }

        The charge unit is elementary charge.

        The parameters and index tuples (corresponding to the atom_idx in self.internal_to_external_indices) are stored as np.ndarrays.
        
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

        if units is None:
            units = {
                     "distance":openmm.unit.nanometer,
                     "angle":openmm.unit.radian,
                     "energy":openmm.unit.kilojoule_per_mole,
                     }


        CHARGE_UNIT = openmm.unit.elementary_charge
        DISTANCE_UNIT = units["distance"]
        ANGLE_UNIT = units["angle"]
        ENERGY_UNIT = units["energy"]

        BOND_EQ_UNIT = DISTANCE_UNIT
        ANGLE_EQ_UNIT = ANGLE_UNIT
        TORSION_K_UNIT = ENERGY_UNIT
        TORSION_PHASE_UNIT = ANGLE_UNIT
        BOND_K_UNIT = ENERGY_UNIT / (DISTANCE_UNIT**2)
        ANGLE_K_UNIT = ENERGY_UNIT / (ANGLE_UNIT**2)



        param_dict = {}
        
        self.graph = self.graph.to("cpu")

        if "n4" in self.graph.ntypes:
            assert self.graph.nodes["n4"].data["k"].shape[1] == self.proper_periodicity
        if self.use_impropers and "n4_improper" in self.graph.ntypes:
            assert self.graph.nodes["n4_improper"].data["k"].shape[1] == self.improper_periodicity



        # ========= CHARGES AND LJ =========
        if self.charge_model is not None:
            self.charges = self.charges_from_model()
        else:
            if self.epsilons is None or self.sigmas is None:
                self.charges = self.charges_from_ff(set_lj=True)
            else:
                self.charges = self.charges_from_ff(set_lj=False)

        if self.epsilons is None or self.sigmas is None:
            self.charges_from_ff(set_lj=True)
        
        assert self.charges is not None
        assert self.sigmas is not None
        assert self.epsilons is not None

        param_dict["atom_q"] = self.charges
        param_dict["atom_sigma"] = self.sigmas
        param_dict["atoms_epsilon"] = self.epsilons

        idxs = np.arange(self.graph.num_nodes("n1"))

        # to external:
        if self.internal_to_external_idx is not None:
            idxs = self.internal_to_external_idx[idxs]
        
        param_dict["atom_idxs"] = idxs


        # ========= BONDS =========
        eqs = self.graph.nodes["n2"].data["eq"].detach().numpy()
        ks = self.graph.nodes["n2"].data["k"].detach().numpy()

        # convert to openmm units:
        eqs = Quantity(eqs, unit=grappa_units.DISTANCE_UNIT).value_in_unit(BOND_EQ_UNIT)
        ks = Quantity(ks, unit=grappa_units.FORCE_CONSTANT_UNIT).value_in_unit(BOND_K_UNIT)

        param_dict["bond_eq"] = eqs
        param_dict["bond_k"] = ks

        idxs = self.graph.nodes["n2"].data["idxs"].detach().numpy()
        if self.internal_to_external_idx is not None:
            idxs = self.internal_to_external_idx[idxs]
        param_dict["bond_idxs"] = idxs


        # ========= ANGLES =========
        if "n3" in self.graph.ntypes:
            eqs = self.graph.nodes["n3"].data["eq"].detach().numpy()
            ks = self.graph.nodes["n3"].data["k"].detach().numpy()

            # convert to openmm units:
            eqs = Quantity(eqs, unit=grappa_units.ANGLE_UNIT).value_in_unit(ANGLE_EQ_UNIT)
            ks = Quantity(ks, unit=grappa_units.ANGLE_FORCE_CONSTANT_UNIT).value_in_unit(ANGLE_K_UNIT)

            param_dict["angle_eq"] = eqs
            param_dict["angle_k"] = ks

            idxs = self.graph.nodes["n3"].data["idxs"].detach().numpy()
            if self.internal_to_external_idx is not None:
                idxs = self.internal_to_external_idx[idxs]
            param_dict["angle_idxs"] = idxs


        # ========= PROPER DIHEDRALS =========
        if "n4" in self.graph.ntypes:
            ns, phases, k_vec = SysWriter.get_periodicity_phase_k(ks=self.graph.nodes["n4"].data["k"])
            
            # convert to openmm units
            phases = Quantity(phases, unit=grappa_units.ANGLE_UNIT).value_in_unit(TORSION_PHASE_UNIT)
            k_vec = Quantity(k_vec, unit=grappa_units.TORSION_FORCE_CONSTANT_UNIT).value_in_unit(TORSION_K_UNIT)

            param_dict["proper_periodicity"] = ns
            param_dict["proper_phase"] = phases
            param_dict["proper_k"] = k_vec

            idxs = self.graph.nodes["n4"].data["idxs"].detach().numpy()
            if self.internal_to_external_idx is not None:
                idxs = self.internal_to_external_idx[idxs]

            param_dict["proper_idxs"] = idxs

        # ========= IMPROPER DIHEDRALS =========
        if not self.use_impropers:
            return param_dict
        
        if not "n4_improper" in self.graph.ntypes:
            return param_dict
        
        ns, phases, k_vec = SysWriter.get_periodicity_phase_k(ks=self.graph.nodes["n4_improper"].data["k"])
        
        # convert to openmm units
        phases = Quantity(phases, unit=grappa_units.ANGLE_UNIT).value_in_unit(TORSION_PHASE_UNIT)
        k_vec = Quantity(k_vec, unit=grappa_units.TORSION_FORCE_CONSTANT_UNIT).value_in_unit(TORSION_K_UNIT)

        param_dict["improper_periodicity"] = ns
        param_dict["improper_phase"] = phases
        param_dict["improper_k"] = k_vec

        idxs = self.graph.nodes["n4_improper"].data["idxs"].detach().numpy()
        if self.internal_to_external_idx is not None:
            idxs = self.internal_to_external_idx[idxs]
        param_dict["improper_idxs"] = idxs

        return param_dict





    def update_system(self)->None:
        """
        Updates the system with parameters from the graph. The epsilons and sigmas are taken from the classical forcefield.
        """

        openmm_units = {
            "distance":openmm.unit.nanometer,
            "angle":openmm.unit.radian,
            "energy":openmm.unit.kilojoule_per_mole,
        }

        param_dict = self.get_parameter_dict(units=openmm_units)

        # ========= SET THE CHARGES =========
        # change the charges to the ones in the charge_dict
        if not self.charge_model is None:
    
            for force in self.sys.getForces():
                if isinstance(force, openmm.NonbondedForce):
                    if force.getNumExceptions() > 0:
                        original_charges = []

                    for i in range(force.getNumParticles()):
                        q_orig, sigma, epsilon = force.getParticleParameters(i)

                        original_charges.append(q_orig.value_in_unit(openmm.unit.elementary_charge))

                        q = self.charges[i]

                        force.setParticleParameters(i, q, sigma, epsilon)
                    for i in range(force.getNumExceptions()):
                        (
                            idx0,
                            idx1,
                            q_exception,
                            sigma,
                            epsilon,
                        ) = force.getExceptionParameters(i)

                        eps = 1e-20

                        # apply the same scaling:
                        ratio_q = self.charges[idx0] * (self.charges[idx1] + eps) / (original_charges[idx0] * original_charges[idx1] + eps)

                        q = ratio_q * q_exception

                        force.setExceptionParameters(
                            i, idx0, idx1, q, sigma, epsilon
                        )


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
                
                
                eqs = param_dict["bond_eq"]
                ks = param_dict["bond_k"]

                # loop through the graph parameters and update the force:
                for i_graph, ((p1,p2), eq, k) in enumerate(zip(self.graph.nodes["n2"].data["idxs"].tolist(), eqs, ks)):

                    # find the index of the bond in the force:
                    idx = self.interaction_indices["n2"][(p1,p2)]
                    force.setBondParameters(idx, particle1=p1, particle2=p2, length=eq, k=k)

            # ========= ANGLES =========
            # assume that each angle only occurs once in the system
            elif isinstance(force, openmm.HarmonicAngleForce):

                if force.getNumAngles() == 0:
                    continue
                if force.getNumAngles() != self.graph.num_nodes("n3"):
                    raise ValueError("Number of angles in the system does not match the number of angles in the graph.")

                eqs = param_dict["angle_eq"]
                ks = param_dict["angle_k"]

                # loop through the graph parameters and update the force:
                for i_graph, ((p1,p2,p3), eq, k) in enumerate(zip(self.graph.nodes["n3"].data["idxs"].tolist(), eqs, ks)):
                    particle1, particle2, particle3 = (p1,p2,p3)
                    # find the index of the bond in the force:
                    idx = self.interaction_indices["n3"][(p1,p2,p3)]
                    force.setAngleParameters(idx, particle1=p1, particle2=p2, particle3=p3, angle=eq, k=k)

            # ========= PROPER AND IMPROPER TORSIONS =========
            elif isinstance(force, openmm.PeriodicTorsionForce):
                if force.getNumTorsions() == 0:
                    continue

                levels = ["n4", "n4_improper"] if self.use_impropers else ["n4"]

                for level in levels:

                    if level not in self.graph.ntypes:
                        continue
                    
                    if level == "n4":
                        key = "proper"
                    elif level == "n4_improper":
                        key = "improper"

                    ns, phases, k_vec = param_dict[f"{key}_periodicity"], param_dict[f"{key}_phase"], param_dict[f"{key}_k"]

                    idxs = self.graph.nodes[level].data["idxs"].tolist()

                    for i_graph, (p1,p2,p3,p4) in enumerate(idxs):

                        for periodicity, phase, k in zip(ns[i_graph], phases[i_graph], k_vec[i_graph]):
                            key = (p1,p2,p3,p4) + (periodicity,)
                            particle1, particle2, particle3, particle4 = (p1,p2,p3,p4)

                            # if this torsion was in the original system, update it
                            if key in self.interaction_indices[level].keys():
                                idx = self.interaction_indices[level][key]
                                force.setTorsionParameters(idx, particle1=particle1, particle2=particle2, particle3=particle3, particle4=particle4, periodicity=periodicity, phase=phase, k=k)
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
    
        total_energies, total_gradients = SysWriter.get_energies_(simulation, np.array(xyz))
        
        
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
        nonbonded_energies, nonbonded_gradients = SysWriter.get_energies_(simulation, np.array(xyz))


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