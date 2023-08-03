#%%
import os

import dgl

from .xyz2res.deploy import xyz2res
from pathlib import Path
import tempfile
from openmm.app import PDBFile, ForceField, Simulation
from typing import List, Union, Tuple, Dict, Callable
from dgl import graph, add_reverse_edges
from openmm.unit import angstrom, unit, kilocalorie_per_mole, Quantity, hartrees, bohr, nanometer
import numpy as np
from .matching import matching
import torch
from .utils import utilities, utils, draw_mol
from ..ff_utils.classical_ff import parametrize, collagen_utility


from ..ff_utils.create_graph import utils as create_graph_utils, graph_init, read_pdb

import matplotlib.pyplot as plt

import math
from .matching import match_utils

from grappa import units as grappa_units


# NOTE: STORE EVERYTHING IN THE GRAPH_DATA DICT AND MAKE self.energies... A PROPERTY
#%%


class PDBMolecule:
    '''
    Class for storing a dataset (positions, energies, forces/gradients) for a molecule that can be described by a pdb file.
    Can be initialized from positions and elements alone, the residues and membership of constituent atoms are deduced in this case.
    Given a sequence and assuming that the atoms are already ordered by residue, can also be initialised more cheaply from a gaussian log file.

    This class can also be used to create PDB files from xyz coordinates and elements alone. See examples/from_xyz.py for a tutorial.

    Note that the order of the stored element/xyz/gradient array may deviate from the element/xyz/gradient array used for initialization to correspond to the order used in the pdb file. To add data manually after initialization, apply the permutation stored by the class to your data beforehand.
    ...

    Attributes
    ----------
    xyz: np.ndarray
        Array of shape (N_conf x N_atoms x 3) containing the atom positions. The order along the N_atoms axis is determined by the order of the elements member variable. This ordering can deviate from the ordering in the initialization arguments.

    energies: np.ndarray
        Array of shape (N_conf) containing the energies in kcal/mol.

    gradients: np.ndarray
        Array of shape (N_conf x N_atoms x 3) containing the gradient of the energy wrt the positions (i.e. the negative forces). The order along the N_atoms axis is determined by the order of the elements member variable. This ordering can deviate from the ordering in the initialization arguments. Unit is kcal/mol/Angstrom.

    elements: np.ndarray
        Array of shape (N_atoms) containing the atomic numbers of the elements. The order can deviate from the ordering in the initialization arguments but corresponds to the stored values for xyz and the gradients.

    pdb: list
        List of strings representing the pdb-file of some conformation.
        This stores information on connectivity and atom types of the molecule, not the on the conformations.

    sequence: str
        String of threeletter amino acid codes representing the sequence of the molecule. Separated by '-', capitalized and including cap residues.

    residues: np.ndarray
        List of strings containing the residues by atom in the order of the element member variable, only set if initialized by xyz.

    residues_numbers: np.ndarray
        List of integers starting at zero and ending at len(sequence), grouping atoms to residues, only set if initialized by xyz.

    name: str
        Name of the molecule, by default, this is the sequence. However this is only unique up to charges and HIS tautomers.

    permutation: np.ndarray
        Array of integers describing an atom permutation representing the change of order of the input atom order (which is done to be consistent with the pdb). This can be used to add own xyz or gradient data after initialization.


    Methods
    -------
    __init__():
        Sets every member to None. Use the from_... classmethods for initialization.

    PDBMolecule.from_gaussian_log(logfile, cap:bool=True, rtp_path=DEFAULT_RTP, sequence:list=None):
        Use a gaussian logfile for initialization.

    PDBMolecule.from_xyz(cls, xyz:np.ndarray, elements:np.ndarray, energies:np.ndarray=None, gradients:np.ndarray=None, rtp_path=DEFAULT_RTP, residues:list=None, res_numbers:list=None):
        Use an xyz array of shape (N_confsxN_atomsx3) and an element array of shape (N_atoms) for initialization. The atom order in which xyz and element are stored may differ from that of those used for initilization (See description of the xyz member).
        The positions must be given in angstrom.
        Currently only works for:
        peptides that do not contain TYR, PHE.
        and R,K: positive charge, D,E: negative charge, H: neutral - HIE


    Usage
    -----
    >>> from PDBData.PDBMolecule import PDBMolecule
    mol = PDBMolecule.from_pdb("my_pdbpath.pdb", xyz=xyz, energies=energies, gradients=gradients)
    mol.bond_check()
    mol.parametrize()
    g = mol.to_dgl()
    '''

    DEFAULT_RTP = Path(__file__).parent.parent/Path("ff_utils/classical_ff/amber99sb-star-ildnp.ff/aminoacids.rtp")

    
    def __init__(self)->None:
        """
        Initializes everything to None. Use the classmethods to construct the object!
        """
        ### replaced by property and setter:
        ########################################
        # self.xyz: np.ndarray = None  #: Array of shape (N_conf x N_atoms x 3) containing the atom positions.
        # self.energies: np.ndarray = None  #: Array of shape (N_conf) containing the energies in kcal/mol.
        # self.gradients: np.ndarray = None  #: Array of shape (N_conf x N_atoms x 3) containing the gradient of the energy wrt the positions.
        ########################################

        self.elements: np.ndarray = None  #: Array of shape (N_atoms) containing the atomic numbers of the elements.

        self.pdb: List[str] = None  #: List of strings representing the pdb-file of some conformation.
        self.sequence: str = None  #: String of threeletter amino acid codes representing the sequence of the molecule. Separated by '-', capitalized and including cap residues.
        self.residues: np.ndarray = None  #: List of strings containing the residues by atom in the order of the element member variable, only set if initialized by xyz.
        self.residue_numbers: np.ndarray = None  #: List of integers starting at zero and ending at len(sequence), grouping atoms to residues, only set if initialized by xyz.
        self.name: str = None  #: Name of the molecule, by default, this is the sequence.
        self.permutation: np.ndarray = None  #: Array of integers describing an atom permutation representing the change of order of the input atom order.

        self.graph_data: Dict = {}  # dictionary holding all of the data in the graph, Graph data is stored by chaining dicts like this: graph_data[node_type][feat_type] = data
        # data is an np.ndarray and forces/positions are of shape (n_atoms, n_confs, 3)

    @property
    def xyz(self):
        """Array of shape (N_conf x N_atoms x 3) containing the atom positions."""
        xyz = self.graph_data["n1"]["xyz"]
        if xyz is None:
            return None
        
        return xyz.transpose(1,0,2)

    @property
    def energies(self):
        """Array of shape (N_conf) containing the energies in kcal/mol."""
        if not "g" in self.graph_data.keys():
            return None
        if not "u_qm" in self.graph_data["g"].keys():
            return None
        en = self.graph_data["g"]["u_qm"]
        if en is None:
            return None
        return en[0,:]

    @property
    def gradients(self):
        """Array of shape (N_conf x N_atoms x 3) containing the gradient of the energy wrt the positions."""
        if not "n1" in self.graph_data.keys():
            return None
        if not "grad_qm" in self.graph_data["n1"].keys():
            return None
        grad = self.graph_data["n1"]["grad_qm"]
        if grad is None:
            return None
        return grad.transpose(1,0,2)

    @property
    def n_atoms(self):
        return len(self.elements)
    
    @property
    def n_confs(self):
        return self.xyz.shape[1]
    

    @xyz.setter
    def xyz(self, value:np.ndarray):
        """Array of shape (N_conf x N_atoms x 3) containing the atom positions."""

        # if not value is None:
        #     assert value.shape[1:] == (self.n_atoms, 3)
        if not "n1" in self.graph_data.keys():
            self.graph_data["n1"] = {}
        if value is None:
            self.graph_data["n1"]["xyz"] = None
        else:
            try:
                self.graph_data["n1"]["xyz"] = value.transpose(1,0,2)
            except:
                print(value.shape)
                raise


    @energies.setter
    def energies(self, value:np.ndarray):
        """Array of shape (N_conf) containing the energies in kcal/mol."""

        # if "n1" in self.graph_data.keys() and not value is None:
        #     assert len(value) == self.n_confs
        if not "g" in self.graph_data.keys():
            self.graph_data["g"] = {}
        if value is None:
            self.graph_data["g"]["u_qm"] = None
        else:
            self.graph_data["g"]["u_qm"] = value[None,:]


    @gradients.setter
    def gradients(self, value:np.ndarray):
        """Array of shape (N_conf x N_atoms x 3) containing the gradient of the energy wrt the positions."""

        # if "n1" in self.graph_data.keys() and not value is None:
        #     assert value.shape == (self.n_confs, self.n_atoms, 3)
        if not "n1" in self.graph_data.keys():
            self.graph_data["n1"] = {}
        if value is None:
            self.graph_data["n1"]["grad_qm"] = None
        else:
            self.graph_data["n1"]["grad_qm"] = value.transpose(1,0,2)


    def write_pdb(self, pdb_path:Union[str, Path]="my_pdb.pdb")->None:
        """
        Creates a pdb file at the given path.
        """
        with open(pdb_path, "w") as f:
            f.writelines(self.pdb)


    def __getitem__(self, idx:int)->Tuple[np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None]]:        
        """
        For the state indexed by idx, returns a tuple of (xyz, energy, gradients), where xyz and gradient are arrays of shape (N_atoms, 3) and energy is a scalar value.
        """
        if (not self.energies is None) and (not self.gradients is None):
            return self.xyz[idx], self.energies[idx], self.gradients[idx]
        
        if self.energies is None and not self.gradients is None:
            return self.xyz[idx], None, self.gradients[idx]

        if not self.energies is None and self.gradients is None:
            return self.xyz[idx], self.energies[idx], None

        return self.xyz[idx], None, None
    

    def __len__(self)->int:
        """
        Returns the number of conformations.
        """
        return len(self.xyz)


    def to_dict(self)->dict:
        """
        Create a dictionary holding arrays of the internal data.
        Graph data is stored by using keys that are of the form f'{level} {feature}'.
        """
        arrays = {}
        arrays["elements"] = self.elements
        arrays["pdb"] = np.array(self.pdb)

        if not self.permutation is None:
            arrays["permutation"] = self.permutation
        if not self.sequence is None:
            arrays["sequence"] = np.array((self.sequence))
        if not self.name is None:
            arrays["name"] = np.array((self.name))
        if not self.residues is None:
            arrays["residues"] = self.residues
        if not self.residue_numbers is None:
            arrays["residue_numbers"] = self.residue_numbers


        for level in self.graph_data.keys():
            for feat in self.graph_data[level].keys():
                key = f'{level} {feat}'
                arrays[key] = self.graph_data[level][feat]

        return arrays
    

    @classmethod
    def from_dict(cls, d):
        """
        Internal helper function. Creates a PDBMolecule from a dictionary.
        Assumes that the dictionary is created by the to_dict method, i.e. that the graph data is stored by using keys that are of the form f'{level} {feature}'.
        """
        self = cls()
        self.elements = d["elements"]

        self.pdb = d["pdb"].tolist()

        if "permutation" in d.keys():
            self.permutation = d["permutation"]
        if "sequence" in d.keys():
            self.sequence = str(d["sequence"]) # assume that the entry is given as np array with one element
        if "name" in d.keys():
            self.name = str(d["name"]) # assume that the entry is given as np array with one element
        if "residues" in d.keys():
            self.residues = d["residues"]
        if "residue_numbers" in d.keys():
            self.residue_numbers = d["residue_numbers"]

        for key in d.keys():
            if " " in key:
                level, feat = key.split(" ")
                if not level in self.graph_data.keys():
                    self.graph_data[level] = {}
                self.graph_data[level][feat] = d[key]

        return self
    

    def save(self, filename)->None:
        """
        Creates an npz file with the uncompressed dataset.
        """
        with open(filename, "bw") as f:
            np.savez(f, **self.to_dict())


    def compress(self, filename)->None:
        """
        Creates an npz file with the compressed dataset.
        """
        with open(filename, "bw") as f:
            np.savez_compressed(f, **self.to_dict())


    @classmethod
    def load(cls, filename):
        """
        Initializes the object from an npz file.
        """
        with open(filename, "br") as f:
            return cls.from_dict(np.load(f))


    def to_ase(self, idxs=None)->List:
        """
        Returns an ASE trajectory of the states given by idxs. Default is to return a trajectory containing all states.
        """
        from ase import Atoms

        if idxs is None:
            idxs = np.arange(len(self))

        def get_ase(xyz, elements):
            ase_mol = Atoms(f"N{len(elements)}")
            ase_mol.set_atomic_numbers(elements)
            ase_mol.set_positions(xyz)
            return ase_mol
        
        traj = [get_ase(self.xyz[idx], self.elements) for idx in idxs]
        return traj


    def get_bonds(self, from_pdb:bool=True, collagen=True)->List[Tuple[int, int]]:
        """
        Returns a list of tuples describing the bonds between the atoms where the indices correspond to the order of self.elements, xyz, etc. .
        """
        if from_pdb:
            openmm_mol = self.to_openmm(collagen=collagen)
            # seq = [res.name for res in openmm_mol.topology.residues()]
            # self.sequence = "-".join(seq).upper()
            bonds = [(b[0].index, b[1].index) for b in openmm_mol.topology.bonds()]
            min_bond = np.array(bonds).flatten().min()
            if min_bond != 0:
                bonds = [(b[0]-min_bond, b[1]-min_bond) for b in bonds]
        else:
            bonds = self.get_ase_bonds(idxs=[0])[0]
        return bonds

    def to_graph(self, from_pdb:bool=True)->graph:
        """
        Returns a homogeneous dgl graph with connectivity induced by the pdb.
        """
        bonds = self.get_bonds(from_pdb=from_pdb)
        # reshape the list from n_bx2 to 2xn_b:
        bonds = [[b[i] for b in bonds] for i in [0,1] ]
        # create an undirected graph:
        g = graph(tuple(bonds))
        g = add_reverse_edges(g)
        # torch onehot encoding:
        g.ndata["atomic_number"] = torch.nn.functional.one_hot(torch.tensor(self.elements))
        return g

    def draw(self):
        """
        Draws the molecular graph using networx.
        """
        g = self.to_graph()
        draw_mol(g, show=True)

    def to_dgl(self, classical_ff:Union[ForceField, str]=None, collagen:bool=False, allow_radicals:bool=True, get_charges=None)->dgl.DGLGraph:
        """
        Returns a heterogeneous dgl graph of n-body tuples for forcefield parameter prediction.
        The data stored in the class is keyed by: ("n1","xyz"), ("g","u_qm") ("n1","grad_qm")
        Uses self.graph_data for construction.
        If the Molecule is already parametrised, forcefield, allow_radicals, collagen and get_charges are ignored.
        """
        if collagen:
            classical_ff = collagen_utility.append_collagen_templates(classical_ff)

        if not "n2" in self.graph_data.keys():
            assert classical_ff is not None, "Please provide a forcefield for the parametrization."
            
            g = self.parametrize(forcefield=classical_ff, allow_radicals=allow_radicals, collagen=collagen, get_charges=get_charges)
        
        # in this case we assume that the graph data is already stored in the object and we initialize the graph from this:
        else:

            # initialize graph:
            n2_idxs, n3_idxs, n4_idxs, n4_improper_idxs = [
                self.graph_data[level]["idxs"]
                    if level in self.graph_data.keys()
                else None
                for level in ["n2", "n3", "n4", "n4_improper"]
            ]
            g = graph_init.get_empty_graph(bond_idxs=n2_idxs, angle_idxs=n3_idxs, proper_idxs=n4_idxs, improper_idxs=n4_improper_idxs, use_impropers=True)

            assert g.num_nodes("n1") == len(self.elements), f"Number of nodes in graph ({g.num_nodes()}) does not match number of atoms ({len(self.elements)})"
            assert g.num_nodes("n2") == len(n2_idxs), f"Number of edges in graph ({g.num_edges()}) does not match number of bonds ({len(n2_idxs)})"

            # write all data stored in self.graph_data in the graph:
            for level in self.graph_data.keys():
                for feat in self.graph_data[level].keys():
                    try:
                        if not self.graph_data[level][feat] is None:
                            g.nodes[level].data[feat] = torch.tensor(self.graph_data[level][feat])
                    except Exception as e:
                        name = "None" if self.name is None else self.name
                        raise RuntimeError(f"Could not write {level} {feat} to graph of name {name}") from e

        return g
    

    def parametrize(self, forcefield:Union[ForceField, str]=ForceField('amber99sbildn.xml'), get_charges=None, allow_radicals=False, collagen=False)->dgl.DGLGraph:
        """
        Stores the forcefield parameters and energies/gradients in the graph_data dictionary.
        get_charges is a function that takes a topology and returns a list of charges as openmm Quantities in the order of the atoms in topology.
        Also writes reference data (such as the energy/gradients minus nonbonded) to the graph.
        If the collagen flag is set to True, the collagen forcefield based on amber99sbildn is used instead of the given forcefield.
        If forcefield is a string, we assume that a small molecule forcefield such as gaff-2.11 is being used. In this case, no information on residues is needed for parametrization.
        """

        if collagen:
            # forcefield = collagen_utility.append_collagen_templates(forcefield)
            # NOTE
            forcefield = collagen_utility.get_collagen_forcefield()

        if isinstance(forcefield, str):
            assert len(self.pdb) == 1, "If forcefield is a string, a smiles string must be present in the object."
            [smiles] = self.pdb

            g = graph_init.graph_from_topology(smiles=smiles, topology=None, classical_ff=forcefield, allow_radicals=allow_radicals, radical_indices=None, xyz=self.xyz, qm_energies=self.energies, qm_gradients=self.gradients, get_charges=get_charges)
    
        else:
            top = self.to_openmm().topology
            g = graph_init.graph_from_topology(topology=top, classical_ff=forcefield, allow_radicals=allow_radicals, radical_indices=None, xyz=self.xyz, qm_energies=self.energies, qm_gradients=self.gradients, get_charges=get_charges)

        # write data in own dict:
        for level in g.ntypes:
            if not level in self.graph_data.keys():
                self.graph_data[level] = {}
            for feat in g.nodes[level].data.keys():
                self.graph_data[level][feat] = g.nodes[level].data[feat].detach().numpy()
            
        return g

    

    def to_rdkit_graph(self):
        """
        Returns an openff molecule for representing the graph structure of the molecule, without chemical details such as bond order, formal charge and stereochemistry.
        """
        openmm_mol = self.to_openmm()
        mol = create_graph_utils.openmm2rdkit_graph(openmm_mol.topology)
        return mol


    def to_openmm(self, collagen=True)->PDBFile:
        """
        Returns an openmm molecule containing the pdb member variable.
        """
        with tempfile.TemporaryDirectory() as tmp:
            pdbpath = os.path.join(tmp, 'pep.pdb')
            with open(pdbpath, "w") as pdb_file:
                pdb_file.writelines([line for line in self.pdb])
            openmm_pdb = PDBFile(pdbpath)
        # NOTE: the following is bad to do that because one cannot differentiate between the case where CB has 2 Hs or there are 2 CBs with one H each.
        # however, it is our standard now...
        openmm_pdb = read_pdb.replace_h23_to_h12(openmm_pdb)
        if collagen:
            openmm_pdb.topology = collagen_utility.add_bonds(openmm_pdb.topology, allow_radicals=True)

        openmm_pdb.topology = utils.rename_cap_Hs(topology=openmm_pdb.topology)
        return openmm_pdb

        
    def get_ase_bonds(self, idxs:List[int]=[0], majority_vote=True)->List[List[Tuple[int, int]]]:
        """
        Returns a list of shape len(idxs)*n_bonds of 2-tuples describing the bonds between the atoms where the indices correspond to the order of self.elements, xyz, etc., inferred by ase from the positions and elements only.
        """
        from ase.geometry.analysis import Analysis
        traj = self.to_ase(idxs)
        if majority_vote:
            try:
                connectivities = [Analysis(traj[id]).nl[0].get_connectivity_matrix() for id in idxs]
            except IndexError:
                raise RuntimeError(f"Index out of bounds for get_ase_bonds. Max idx is {max(idxs)} but there are only {len(self)} conformations.")
            bonds = [[(n1,n2) for (n1,n2) in c.keys() if n1!=n2] for c in connectivities]

        else:
            bonds = match_utils.bond_majority_vote(traj)
        return bonds
        

    def validate_confs(self, forcefield=ForceField("amber99sbildn.xml"), collagen:bool=False, quickload:bool=False)->Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Returns the rmse between the classical force field in angstrom and kcal/mol and the stored data, and the standard deviation of the data itself.
        """

        e_class, grad_class = self.get_ff_data(forcefield, collagen=collagen, quickload=quickload)
        e_class -= e_class.mean()
        en = self.energies - self.energies.mean(axis=-1)
        diffs_e = en - e_class
        if not self.gradients is None:
            diffs_g = self.gradients - grad_class
        
        if not self.gradients is None:
            return (np.sqrt(np.mean(diffs_e**2)), np.std(en)), (np.sqrt(np.mean(diffs_g**2)), np.std(self.gradients))
        else:
            return (np.sqrt(np.mean(diffs_e**2)), np.std(en)), (None, None)
        

    def conf_check(self, forcefield=ForceField("amber99sbildn.xml"), sigmas:Tuple[float,float]=(1.,1.), collagen=False, quickload:bool=False)->bool:
        """
        Checks if the stored energies and gradients are consistent with the forcefield. Returns True if the energies and gradients are within var_param[0] and var_param[1] standard deviations of the stored energies/forces.
        If sigmas are 1, this corresponds to demanding that the forcefield data is better than simply always guessing the mean.
        """
        (rmse_e, std_e), (rmse_g, std_g) = self.validate_confs(forcefield, collagen=collagen, quickload=quickload)
        if not self.gradients is None:
            return rmse_e < sigmas[0]*std_e and rmse_g < sigmas[1]*std_g
        else:
            return rmse_e < sigmas[0]*std_e


    def filter_confs(self, max_energy:float=60., max_force:float=None, reference=False, mask=None)->bool:
        """
        Filters out conformations with energies or forces that are over max_energy kcal/mol (or max_force kcal/mol/angstrom) away from the minimum of the dataset (not the actual minimum). Apply this before parametrizing or re-apply the parametrization after filtering. Units are kcal/mol and kcal/mol/angstrom.
        Returns True if more than two conformations remain.
        If reference is true, uses u_ref and grad_ref, i.e. usually the qm value subtracted by the nonbonded contribution instead of the stored energies and gradients. This can only be done after parametrising.
        For filtering the graph data, assume that entries are energies if and only if they are stored in 'g' and have 'u_' in their feature name. For gradients, the same applies with 'n1' and 'grad_'.
        If a mask is provided, this mask is used additionally.
        """
        if not mask is None:
            assert mask.shape[0] == 1, f"mask should be of shape (1,n_confs), i.e. batching is disabled, shape is {mask.shape}"

        if (not max_energy is None) and (not self.energies is None):
            if not reference:
                energies = self.energies - self.energies.min()
            else:
                if len(self.graph_data.keys()) == 0:
                    raise RuntimeError("No graph data found. Please parametrize first or set reference=False.")
                # take the reference energies for filtering
                if not "u_ref" in self.graph_data["g"].keys():
                    raise RuntimeError(f"No reference energies found. Please parametrize first or set reference=False, \ngraph_data keys are {self.graph_data['g'].keys()}")
                energies = self.graph_data["g"]["u_ref"] - self.graph_data["g"]["u_ref"].min()

            # create a mask for energies below max_energy
            energy_mask = np.abs(energies) < max_energy
            # append a zero dimension if it is not there
            if len(energy_mask.shape) == 1:
                energy_mask = energy_mask[None,:]
        else:
            energy_mask = np.ones((1,len(self.xyz)), dtype=bool)

        if (not max_force is None) and (not self.gradients is None):
            if not reference:
                forces = self.gradients
            else:

                if not "grad_ref" in self.graph_data["n1"].keys():
                    raise RuntimeError(f"No reference gradients found. Please parametrize first or set reference=False., \ngraph_data keys are {self.graph_data['n1'].keys()}")
                forces = self.graph_data["n1"]["grad_ref"].transpose(1,0,2)
            
            # for each conf, take the maximum along atoms and spatial dimension

            forces = np.max(np.max(np.abs(forces), axis=-1), axis=-1)

            force_mask = forces < max_force
            # append a zero dimension if it is not there
            if len(force_mask.shape) == 1:
                force_mask = force_mask[None,:]
        else:   
            force_mask = np.ones((1,len(self.xyz)), dtype=bool)

        if mask is None:
            mask = np.ones((1,len(self.xyz)), dtype=bool)

        mask *= energy_mask * force_mask

        assert mask.shape[0] == 1, f"mask should be of shape (1,n_confs), i.e. batching is disabled, shape is {mask.shape}"
        mask = mask[0]
        
        # apply mask:

        for key in self.graph_data.keys():
            if key == "g":
                # graph data for the whole molecule
                for k in self.graph_data[key].keys():
                    if "u_" in k:
                        if self.graph_data[key][k] is None:
                            continue
                        # assume shape of energies is (1,n_confs).
                        try:
                            self.graph_data[key][k] = self.graph_data[key][k][:,mask]
                        except Exception as e:
                            raise RuntimeError(f"Error while filtering energies, key is {k}, shape is {self.graph_data[key][k].shape}, mask shape is {mask.shape}, error is {type(e)}: {e}")

            elif key == "n1":
                for k in self.graph_data[key].keys():
                    if "grad_" in k or "xyz" in k:
                        if self.graph_data[key][k] is None:
                            continue
                        # assume shape of grad and xyz is (atoms,n_confs,3).
                        try:
                            self.graph_data[key][k] = self.graph_data[key][k][:,mask]
                        except Exception as e:
                            raise RuntimeError(f"Error while filtering energies, key is {k}, shape is {self.graph_data[key][k].shape}, mask shape is {mask.shape}, error is {type(e)}: {e}")

        
        # return True if there are more than two conformations left
        return self.xyz.shape[0] > 1


    def bond_check(self, idxs:List[int]=None, perm:bool=False, seed:int=0, collagen=True)->None:
        """
        For each configuration, creates a list of tuples describing the bonds between the atoms where the indices correspond to the order of self.elements, xyz, etc. and compares it to the list of bonds inferred by ase from the positions and elements only. Throws an error if the lists are not identical.
        If perm is True, also creates a list of bonds for a permutation of the atoms and checks if the lists are identical after permutating back.
        """

        # NOTE: what about indistinguishable atoms? (e.g. hydrogens)
        # -> should be fine since they have the same bonds if they are indistuinguishable

        if idxs is None:
            idxs = np.arange(len(self))
        # sort the bonds by the smaller index:
        pdb_bonds = [( min(b[0],b[1]), max(b[0], b[1]) ) for b in self.get_bonds(collagen=collagen)]
        ase_bonds = [[( min(b[0],b[1]), max(b[0], b[1]) ) for b in id_bonds] for id_bonds in self.get_ase_bonds(idxs)]

        atoms = [a.name for a in self.to_openmm(collagen=collagen).topology.atoms()]

        for id in idxs:
            ona = set(pdb_bonds) - set(ase_bonds[id])
            ano = set(ase_bonds[id]) - set(pdb_bonds)
            if len(ona) > 0 or len(ano) > 0:
                atoms = [a.name for a in self.to_openmm().topology.atoms()]

                errstr = f"Encountered dissimilar bonds during PDBMolecule.bond_check for config {id}.\nIn openmm but not ASE: {ona}, atoms {[(atoms[i], atoms[j]) for (i,j) in ona]}.\nIn ASE but not openmm: {ano}, atoms {[(atoms[i], atoms[j]) for (i,j) in ano]}. (The indices are starting at zero not at one!)"
                raise RuntimeError(errstr)

        if perm:

            # check if the bonds are the same after permuting the atoms:

            # get the permutation:
            np.random.seed(seed)
            perm = np.arange(len(self.elements))
            perm = np.random.permutation(perm)

            # permute the atoms:
            new_xyz = self.xyz[:,perm]
            new_elements = self.elements[perm]
            # create a new molecule:
            new_mol = PDBMolecule.from_xyz(xyz=new_xyz, elements=new_elements)
            # the new mol performed a permutation of the input, so we need to chain the two permutations:
            perm = perm[new_mol.permutation]
            
            # get the bonds:
            # [print(b[0], b[1]) for b in new_mol.to_openmm().topology.bonds()]

            new_bonds = new_mol.get_bonds(from_pdb=True)

            # permute indices back:
            perm_map = {i:np.arange(len(self.elements))[perm][i] for i in range(len(perm))}

            assert np.all([self.xyz[:,perm_map[i]]==new_mol.xyz[:,i] for i in range(len(self.elements))])

            # sort the bonds by the smaller index:
            new_bonds = [(perm_map[b[0]], perm_map[b[1]]) for b in new_bonds]
            new_bonds = [( min(b[0], b[1]), max(b[0], b[1]) ) for b in new_bonds]

            # check if the bonds are the same after permutating back:

            if set(pdb_bonds) != set(new_bonds):
                # find the bonds that are in one but not the other:
                nmo = set(new_bonds) - set(pdb_bonds)
                omn = set(pdb_bonds) - set(new_bonds)
                atoms = [a.name for a in self.to_openmm().topology.atoms()]

                errstr = f"Encountered dissimilar bonds during PDBMolecule.bond_check for permutated version.\nIn permuted but not original: {nmo}, atoms {[(atoms[i], atoms[j]) for (i,j) in nmo]}.\noriginal but not permuted {omn}, atoms {[(atoms[i], atoms[j]) for (i,j) in omn]}. (The indices are starting at zero not at one and belong to the original version!)"
                raise RuntimeError(errstr)


    @classmethod
    def from_pdb(cls, pdbpath:Union[str,Path], xyz:np.ndarray=None, energies:np.ndarray=None, gradients:np.ndarray=None):
        """
        Initializes the object from a pdb file and an xyz array of shape (N_confsxN_atomsx3). Units must be the ones specified for the class. The atom order is the same as in the pdb file.
        """

        pdbpath = str(pdbpath)

        for l, to_be_checked, name in [(1,energies, "energies"), (3,gradients,"gradients")]:
            if not to_be_checked is None:
                if xyz.shape[0] != to_be_checked.shape[0]:
                    raise RuntimeError(f"Number of conformations must be consistent between xyz and {name}, shapes are {xyz.shape} and {to_be_checked.shape}")
                if len(to_be_checked.shape) != l:
                    raise RuntimeError(f"{name} must be a {l}-dimensional array but is of shape {to_be_checked.shape}")
                
        if not gradients is None:
            if xyz.shape != gradients.shape:
                raise RuntimeError(f"Gradients and positions must have the same shape. Shapes are {gradients.shape} and {xyz.shape}.")
            
        self = cls()
        pdbmol = PDBFile(pdbpath)
        sequence = "-".join([res.name for res in pdbmol.topology.residues()]).upper()
        self.sequence = sequence
        self.name = self.sequence
        self.elements = np.array([a.element.atomic_number for a in pdbmol.topology.atoms()])
        if xyz is None:
            self.xyz = pdbmol.getPositions(asNumpy=True).value_in_unit(angstrom).reshape(1,-1,3)
        else:
            self.xyz = xyz

        with open(pdbpath, "r") as f:
            self.pdb = f.readlines()

        self.energies = energies
        self.gradients = gradients
        self.permutation = np.arange(len(self.elements))
        
        # NOTE: implement a way to get the sequence from the pdb file?

        return self

    @classmethod
    def from_gaussian_log(cls, logfile:Union[str,Path], cap:bool=True, rtp_path=None, sequence:list=None, logging:bool=False, e_unit:unit=kilocalorie_per_mole*23.0609, dist_unit=angstrom, force_unit=kilocalorie_per_mole*23.0609/angstrom):
        # assume by default that the energy unit is eV and the distance unit is angstrom
        """
        Use a gaussian logfile for initialization. Returns the initialized object.
        By default, it is assumed that the energy unit is eV and the distance unit is angstrom
        Parameters
        ----------
        logfile: str/pathlib.Path
            Path to the gaussian log file
        """

        from ase.calculators.calculator import PropertyNotImplementedError

        if rtp_path is None:
            rtp_path = PDBMolecule.DEFAULT_RTP


        rtp_path = Path(rtp_path)

        self = cls()

        AAs_reference = matching.read_rtp(rtp_path)

        logfile = Path(logfile)

        if sequence is None:
            sequence = matching.seq_from_filename(logfile, AAs_reference, cap)

        self.sequence = ''
        for aa in sequence:
            self.sequence += aa.upper()
            self.sequence += '-'
        self.sequence = self.sequence[:-1]

        self.name = self.sequence

        mol, trajectory = matching.read_g09(logfile, sequence, AAs_reference, log=logging)

        atom_order = matching.match_mol(mol, AAs_reference, sequence, log=logging)
        self.permutation = np.array(atom_order)

        # write single pdb
        conf = trajectory[0]
        self.pdb, elements = PDBMolecule.pdb_from_ase(ase_conformation=conf, sequence=sequence, AAs_reference=AAs_reference,atom_order=atom_order)

        self.elements = np.array(elements)

        # write xyz and energies
        energies = []
        positions = []
        gradients = []

        # NOTE: take higher precision for energies since we are interested in differences that are tiny compared to the absolute magnitude
        for step in trajectory:
            if not energies is None:
                try:
                    energy = step.get_total_energy()
                    energies.append(energy) 
                except PropertyNotImplementedError:
                    energies = None

            if not gradients is None:
                try:
                    grad = -step.get_forces()
                    gradients.append(grad)
                except PropertyNotImplementedError:
                    gradients = None

            pos = step.positions # array of shape n_atoms x 3 in the correct order
            positions.append(pos)
        
        if not energies is None:
            energies = np.array(energies)
            energies -= energies.min()
            self.energies = Quantity(energies, unit=e_unit).value_in_unit(kilocalorie_per_mole)
        else:
            self.energies = None

        if not gradients is None:
            # apply permutation to gradients
            gradients = np.array(gradients)[:,atom_order]
            self.gradients = Quantity(gradients, unit=force_unit).value_in_unit(kilocalorie_per_mole/angstrom)
        else:
            self.gradients = None

        # apply permutation to positions
        positions = np.array(positions)[:,atom_order]
        self.xyz = Quantity(positions, unit=dist_unit).value_in_unit(angstrom)
        
        return self
    
    @classmethod
    def from_gaussian_log_rad(cls, logfile:Union[str,Path], cap:bool=True, rtp_path=None, logging:bool=False, e_unit:unit=kilocalorie_per_mole*23.0609, dist_unit=angstrom, force_unit=kilocalorie_per_mole*23.0609/angstrom):
        # assume by default that the energy unit is eV and the distance unit is angstrom
        """
        Use a gaussian logfile for initialization. Returns the initialized object.
        By default, it is assumed that the energy unit is eV and the distance unit is angstrom
        Parameters
        ----------
        logfile: str/pathlib.Path
            Path to the gaussian log file

        # it may occur that the carbon atom types could not be matched to the correct number, e.g. CG1 in ILE has 2Hs and CG2 has 3Hs. In a radical, it can occur that both have 2 Hs. Problem: the nomenclature is different between the two, it is HG21 HG22 HG23 for CG2 and HG12 HG13 for CG1. So the Hs are not matched correctly. In this case, we just take the first H in the reference template. Thus, matching might make problems.
        """

        from ase.calculators.calculator import PropertyNotImplementedError

        if rtp_path is None:
            rtp_path = PDBMolecule.DEFAULT_RTP

        rtp_path = Path(rtp_path)
        logfile = Path(logfile)

        self = cls()

        AAs_reference, sequence = matching.get_radref(rtp_path=rtp_path, filename=logfile, cap=cap)
        
        self.sequence = ''
        for aa in sequence:
            self.sequence += aa.upper()
            self.sequence += '-'
        self.sequence = self.sequence[:-1]

        self.name = self.sequence

        mol, trajectory = matching.read_g09(logfile, sequence, AAs_reference, log=logging)

        atom_order = matching.match_mol(mol, AAs_reference, sequence, log=logging)
        self.permutation = np.array(atom_order)

        # write single pdb
        conf = trajectory[0]
        self.pdb, elements = PDBMolecule.pdb_from_ase(ase_conformation=conf, sequence=sequence, AAs_reference=AAs_reference,atom_order=atom_order)

        self.elements = np.array(elements)

        # write xyz and energies
        energies = []
        positions = []
        gradients = []

        # NOTE: take higher precision for energies since we are interested in differences that are tiny compared to the absolute magnitude
        for step in trajectory:
            if not energies is None:
                try:
                    energy = step.get_total_energy()
                    energies.append(energy) 
                except PropertyNotImplementedError:
                    energies = None

            if not gradients is None:
                try:
                    grad = -step.get_forces()
                    gradients.append(grad)
                except PropertyNotImplementedError:
                    gradients = None

            pos = step.positions # array of shape n_atoms x 3 in the correct order
            positions.append(pos)
        
        if not energies is None:
            energies = np.array(energies)
            energies -= energies.min()
            self.energies = Quantity(energies, unit=e_unit).value_in_unit(kilocalorie_per_mole)

        if not gradients is None:
            # apply permutation to gradients
            gradients = np.array(gradients)[:,atom_order]
            self.gradients = Quantity(gradients, unit=force_unit).value_in_unit(kilocalorie_per_mole/angstrom)

        # apply permutation to positions
        positions = np.array(positions)[:,atom_order]
        self.xyz = Quantity(positions, unit=dist_unit).value_in_unit(angstrom)

        if "ILE_R" in self.sequence:
            if any(["HG11" in line for line in self.pdb]):
                raise RuntimeError("ILE_R has HG11 in the pdb file. This is not allowed. Please check the pdb file. it may occur that the carbon atom types could not be matched to the correct number, e.g. CG1 in ILE has 2Hs and CG2 has 3Hs. In a radical, it can occur that both have 2 Hs. Problem: the nomenclature is different between the two, it is HG21 HG22 HG23 for CG2 and HG12 HG13 for CG1. So the Hs are not matched correctly. In this case, we just take the first H in the reference template. Thus, matching might make problems.")
            
        return self

    @classmethod
    def from_xyz(cls, xyz:np.ndarray, elements:np.ndarray, energies:np.ndarray=None, gradients:np.ndarray=None, rtp_path=None, residues:list=None, res_numbers:list=None, logging:bool=False, debug:bool=False, smiles:str=None):
        """
        Use an xyz array of shape (N_confsxN_atomsx3) and an element array of shape (N_atoms) for initialization. The atom order in which xyz and element are stored may differ from that of those used for initilization (See description of the xyz member). Units must be those specified for the class (angstrom, kcal/mol, kcal/mol/angstrom)
        Currently only works for the standard amino acids:
        R,K: positive charge, D,E: negative charge, H: neutral - HIE

        If initialised with a smile, no pdb file is created, instead, the smile string is stored in self.pdb.
        """

        from ase import Atoms

        for l, to_be_checked, name in [(1,energies, "energies"), (3,gradients,"gradients")]:
            if not to_be_checked is None:
                if xyz.shape[0] != to_be_checked.shape[0]:
                    raise RuntimeError(f"Number of conformations must be consistent between xyz and {name}, shapes are {xyz.shape} and {to_be_checked.shape}")
                if len(to_be_checked.shape) != l:
                    raise RuntimeError(f"{name} must be a {l}-dimensional array but is of shape {to_be_checked.shape}")
                
        if not gradients is None:
            if xyz.shape != gradients.shape:
                raise RuntimeError(f"Gradients and positions must have the same shape. Shapes are {gradients.shape} and {xyz.shape}.")
            
        if not elements.shape[0] == xyz.shape[1]:
            raise RuntimeError(f"Number of atoms in xyz and elements must be the same. Shapes are {xyz.shape} and {elements.shape}.")
        
        if not smiles is None:
            self = cls()
            self.pdb = [smiles]
            self.name = smiles
            self.sequence = smiles
            self.elements = elements
            self.xyz = xyz
            self.energies = energies
            self.gradients = gradients
            self.permutation = np.arange(len(self.elements))
            return self

        if rtp_path is None:
            rtp_path = PDBMolecule.DEFAULT_RTP

        rtp_path = Path(rtp_path)

        self = cls()
        
        # get the residues:
        if residues is None or res_numbers is None:
            residues, res_numbers = xyz2res(xyz[0], elements, debug=debug)


        # order such that residues belong toghether 
        l = [(res_numbers[i], i) for i in range(len(res_numbers))]
        if not all(l[i] <= l[i+1] for i in range(len(l) - 1)):
            l.sort()

        perm = [l[i][1] for i in range(len(l))]

        xyz = xyz[:,perm]
        elements = elements[perm]
        if not gradients is None:
            gradients = gradients[:,perm]

        if not energies is None:
            self.energies = energies

        # make the sequence: first order the residues according to the ordering above
        residues = np.array(residues)[perm]
        res_numbers = np.array(res_numbers)[perm]

        # match_mol will leave this invariant, therefore we can set it already
        self.res_numbers = np.array(res_numbers)
        self.residues = residues
        
        seq = []
        num_before = -1
        for j,i in enumerate(res_numbers):
            if i != num_before:
                num_before = i
                seq.append(residues[j])

        # preparation for getting the pdb file
        AAs_reference = matching.read_rtp(rtp_path)

        # generate ase molecule from permuted xyz and elements:
        n = xyz[0].shape[0]
        pos = xyz[0]
        ase_mol = Atoms(f"N{n}")
        ase_mol.set_positions(pos)

        ase_mol.set_atomic_numbers(elements)

        mol, _ = matching.read_g09(None, seq, AAs_reference, trajectory_in=[ase_mol], log=logging)

        # this leaves the sequence invariant.
        atom_order = matching.match_mol(mol, AAs_reference, seq, log=logging)

        # concatenate the permutations:

        pdb, _ = PDBMolecule.pdb_from_ase(ase_conformation=ase_mol, sequence=seq, AAs_reference=AAs_reference, atom_order=atom_order)

        # NOTE convert to openmm standard:
        # with tempfile.NamedTemporaryFile(mode='r+') as f:
        # with open("t.pdb",mode='r+') as f:
        #     f.writelines(pdb)
        #     openmm_pdb = PDBFile(f.name)
        #     PDBFile.writeModel(topology=openmm_pdb.getTopology(), positions=xyz[0], file=f)
        #     pdb = f.readlines()
        self.pdb = pdb

        # after applying the perm from xyz2res, we also apply the permutation from the matching:

        self.elements = elements[atom_order]
        self.permutation = np.array(perm)[atom_order]
        self.xyz = xyz[:,atom_order]
        if not gradients is None:
            self.gradients = gradients[:,atom_order]


        self.sequence = ''
        for aa in seq:
            self.sequence += aa.upper()
            self.sequence += '-'
        self.sequence = self.sequence[:-1]

        self.name = self.sequence        

        return self


    @staticmethod
    def pdb_from_ase(ase_conformation, sequence, AAs_reference, atom_order):
        """
        Helper method for internal use. Generates a pdb file from an ase molecule, permuting the atoms in the ase mol according to atom_order.
        """
        pdb_lines = []
        elements = []

        # copied from PDBMolecule.write_trjtopdb
        AA_pos = 0
        elements = ase_conformation.get_atomic_numbers()[atom_order]
        res_len = len(AAs_reference[sequence[AA_pos]]['atoms'])
        for j,k in enumerate(atom_order):
            if j >= res_len:
                AA_pos += 1
                res_len += len(AAs_reference[sequence[AA_pos]]['atoms'])
            atomname = AAs_reference[sequence[AA_pos]]['atoms'][j-res_len][0]

            pdb_lines.append('{0}  {1:>5d} {2:^4s} {3:<3s} {4:1s}{5:>4d}    {6:8.3f}{7:8.3f}{8:8.3f}{9:6.2f}{10:6.2f}          {11:>2s}\n'.format('ATOM',j,atomname ,sequence[AA_pos].split(sep="_")[0],'A',AA_pos+1,*ase_conformation.positions[k],1.00,0.00,atomname[0]))

        pdb_lines.append("END")
        return pdb_lines, elements
    




    def get_ff_data(self, forcefield:Union[ForceField, Callable, str]=None, collagen:bool=False, quickload:bool=False)->Tuple[np.ndarray, np.ndarray]:
        """
        Returns energies, gradients that are calculated by the forcefield for all states xyz. Tha state axis is the zeroth axis.
        If str, assume that it is a small molecule forcefield.
        quickload: simply take the reference energies written in the graph upon parametrization.
        """
        from openmm.app import ForceField, PDBFile, topology, Simulation, PDBReporter, PME, HBonds, NoCutoff, CutoffNonPeriodic
        from openmm import LangevinMiddleIntegrator, Vec3
        from openmm.unit import Quantity, picosecond, kelvin, kilocalorie_per_mole, picoseconds, angstrom, nanometer, femtoseconds, newton
        from grappa import units as grappa_units

        assert not self.xyz is None
        assert not self.elements is None
        assert not self.pdb is None

        if forcefield is None:
            # compare with reference ff
            quickload = True

        if quickload:
            if "u_total_ref" in self.graph_data["g"].keys() and "grad_total_ref" in self.graph_data["n1"].keys():
                e_class = self.graph_data["g"]["u_total_ref"]
                grad_class = self.graph_data["n1"]["grad_total_ref"].transpose(1,0,2)
            return e_class, grad_class

        
        if collagen:
            forcefield = collagen_utility.append_collagen_templates(forcefield)

        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 1*femtoseconds)

        if isinstance(forcefield, str):
            from grappa.ff_utils.SysWriter import SysWriter

            assert len(self.pdb) == 1
            [smiles] = self.pdb

            writer = SysWriter.from_smiles(smiles)
            system = writer.sys
            top = writer.top

        else:
            top = self.to_openmm(collagen=collagen).topology
            system = forcefield.createSystem(top)


        simulation = Simulation(
            top, system=system, integrator=integrator
        )

        return PDBMolecule.get_data_from_simulation(simulation=simulation, xyz=self.xyz)
    


    @staticmethod
    def get_data_from_simulation(simulation:Simulation, xyz:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
        """
        Returns energies, gradients that are calculated by the forcefield for all states xyz. The state axis is the zeroth axis.
        """

        ff_forces = []
        ff_energies = []
        for pos in xyz:
            pos = Quantity(pos, unit=grappa_units.DISTANCE_UNIT).value_in_unit(nanometer)
            simulation.context.setPositions(pos)
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

    @staticmethod
    def do_energy_checks(path:Path, seed:int=0, permute:bool=True, n_conf:int=5, forcefield=None, accuracy:List[float]=[0.1, 1.], verbose:bool=True, fig:bool=True)->bool:
        """
        For all .pdb files in the path, create a set of configurations using openmm, then shuffle positions, elements and forces, create a PDBMolecule and use openmm to calculate the energies and forces belonging to the (re-ordered) positions of the mol, and compare these elementwise. Returns false if the energies deviate strongly.
        accuracy: the maximum allowed deviations in kcal/mol and kcal/mol/angstrom
        """
        counter = 0
        crashed = []

        en_diffs = []

        if verbose:
            print("doing energy checks...")
    
        for p in path.rglob('*.pdb'):
            if verbose:
                print(counter, end='\r')
            counter += 1
            pdb = PDBFile(str(p))
            l = []
            if not PDBMolecule.energy_check(pdb, seed=seed, permute=permute, n_conf=n_conf, forcefield=forcefield, accuracy=accuracy, store_energies_ptr=l):
                # get residue names:
                residues = list(set([a.residue.name for a in pdb.topology.atoms()]))
                crashed.append(residues)
                en_diffs.append(np.mean(np.abs(l[0]-l[1])))

        crashcount = {}

        if len(crashed) > 0:
            import matplotlib.pyplot as plt
            for l in crashed:
                for res in l:
                    if not res in crashcount.keys():
                        crashcount[res] = 1
                    else:
                        crashcount[res] += 1
            if "ACE" in crashcount.keys():
                crashcount.pop("ACE")
            if "NME" in crashcount.keys():
                crashcount.pop("NME")
            if verbose:
                print()
                print(f"energy check errors occured {len(crashed)} out of {counter} times.\nThe following residues where involved:\n{crashcount}")
                print(f"mean energy difference: {np.mean(np.array(en_diffs))} kcal/mol")
                if fig and len(crashed) > 1:
                    plt.figure(figsize=(7,4))
                    plt.title("Failed energy test occurences")
                    plt.bar(crashcount.keys(), crashcount.values())
                    plt.savefig(f"failed_energy_{path.stem}.png")

                    plt.figure(figsize=(7,4))
                    plt.title("Energy diffs")
                    plt.hist(np.array(en_diffs), bins=20)
                    plt.savefig(f"energy_diffs_{path.stem}.png")
            return False
        else:
            if verbose:
                print()
                print(f"no errors occured for {counter} files")
            return True

                

    def energy_check(self, pdb:PDBFile=None, seed:int=0, permute:bool=True, n_conf:int=5, forcefield:ForceField=ForceField('amber99sbildn.xml'), accuracy:List[float]=[0.1, 1.], store_energies_ptr:list=None, collagen=False)->bool:
        """
        Create a set of configurations using openmm, then shuffle positions, elements and forces, create a PDBMolecule from the xyz and elements alone and use openmm to calculate the energies and forces belonging to the (re-ordered) positions of the mol, and compare these elementwise. Returns false if the energies deviate strongly.
        This can be used as static method applied to an uninitialised class object by specifying the pdb file, or as a method of a PDBMolecule object.
        accuracy: the maximum allowed deviations in kcal/mol and kcal/mol/angstrom
        """
        if pdb is None:
            pdb = self.to_openmm(collagen=collagen)
        from openmm.app import ForceField, PDBFile, topology, Simulation, PDBReporter, PME, HBonds, NoCutoff, CutoffNonPeriodic
        from openmm import LangevinMiddleIntegrator, Vec3
        from openmm.unit import Quantity, picosecond, kelvin, kilocalorie_per_mole, picoseconds, angstrom, nanometer, femtoseconds, newton

        if collagen:
            forcefield = collagen_utility.append_collagen_templates(forcefield)

        # generate a dataset

        integrator = LangevinMiddleIntegrator(500*kelvin, 1/picosecond, 0.5*femtoseconds)

        system = forcefield.createSystem(pdb.topology)

        simulation = Simulation(
            pdb.topology, system=system, integrator=integrator
        )

        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy(maxIterations=int(10**3), tolerance=accuracy[0]*kilocalorie_per_mole)
        simulation.step(100)

        elements = np.array([a.element.atomic_number for a in pdb.topology.atoms()])

        amber_e = []
        amber_f = []
        xyz = []
        for t in range(n_conf):
            simulation.step(100)
            state = simulation.context.getState(
                    getEnergy=True,
                    getPositions=True,
                    getForces=True,
                )
            e = state.getPotentialEnergy()
            e = e.value_in_unit(kilocalorie_per_mole)
            f = state.getForces(True)
            f = f.value_in_unit(kilocalorie_per_mole/angstrom)
            f = np.array(f)
            pos = state.getPositions().value_in_unit(angstrom)

            xyz.append(np.array(pos))
            amber_e.append(e)
            amber_f.append(f)

        amber_e = np.array(amber_e)
        amber_f = np.array(amber_f)
        xyz = np.array(xyz)

        # shuffle the atoms
        perm = np.arange(xyz.shape[1])
        np.random.seed(seed=seed)
        if permute:
            np.random.shuffle(perm)
        xyz = xyz[:,perm]
        elements = elements[perm]
        amber_f = amber_f[:,perm]

     # generate a pdb molecule with our package
        mol = PDBMolecule.from_xyz(xyz=xyz, elements=elements, energies=amber_e, gradients=-amber_f)

        new_energies, new_grad = mol.get_ff_data(forcefield=forcefield, collagen=collagen)

        emse, fmse = np.sum((new_energies - mol.energies)**2)/len(new_energies), np.sum((new_grad - mol.gradients)**2)/len(new_grad)

        if not store_energies_ptr is None:
            store_energies_ptr.extend([new_energies, mol.energies, new_grad, mol.gradients])

        if emse > accuracy[0] or fmse > accuracy[1]:
            return False
        else:
            return True
        

    def compare_with_ff(self, ff=None, fontsize:float=16, ff_title:str="Forcefield", compare_ref:bool=False)->plt.axes:
        """
        Calculates energies and forces from the forcefield provided for the conformations stored.
        Returns a plt axes object containing a scatter plot of the energies and forces.
        """
        if ff is None:
            compare_ref = True

        assert not self.gradients is None, "gradients not set"

        energies, forces = self.get_ff_data(ff, quickload=compare_ref)
        energies -= energies.mean()
        self_energies = self.energies - self.energies.min()

        fig, ax = plt.subplots(1,2, figsize=(10,5))

        ax[0].scatter(self_energies, energies)
        ax[0].plot(self_energies, self_energies, color="black", linestyle="--")
        ax[0].set_title("Energies [kcal/mol]]", fontsize=fontsize)
        ax[0].set_xlabel("QM Energies", fontsize=fontsize)
        ax[0].set_ylabel(f"{ff_title} Energies", fontsize=fontsize)
        ax[0].tick_params(axis='both', which='major', labelsize=fontsize-2)

        ax[1].scatter(self.gradients.flatten(), forces.flatten(), s=1, alpha=0.4)
        ax[1].plot(self.gradients.flatten(), self.gradients.flatten(), color="black", linestyle="--")
        ax[1].set_xlabel("QM forces", fontsize=fontsize)
        ax[1].set_ylabel(f"{ff_title} forces", fontsize=fontsize)
        ax[1].set_title("Forces [kcal/mol/Å]", fontsize=fontsize)
        ax[1].tick_params(axis='both', which='major', labelsize=fontsize-2)

        rmse_energies = np.sqrt(np.mean((energies - self_energies)**2))

        rmse_forces = np.sqrt(np.mean((forces - self.gradients)**2))

        ax[0].text(0.05, 0.95, f"RMSE: {rmse_energies:.2f} kcal/mol", transform=ax[0].transAxes, fontsize=fontsize-2, verticalalignment='top')


        ax[1].text(0.05, 0.95, f"RMSE: {rmse_forces:.2f} kcal/mol/Å", transform=ax[1].transAxes, fontsize=fontsize-2, verticalalignment='top')

        plt.tight_layout()
        
        return fig, ax

    @classmethod
    def get_example(cls):
        """
        Returns an example PDBMolecule object with 50 conformations.
        """
        return cls.load(Path(__file__).parent/Path("example_PDBMolecule.npz"))
    

    ###########################################################################################
    # def compare_with_espaloma(self, tag:str="latest"):
    #     """
    #     Write espaloma energies and gradients in self.graph_data.
    #     """
    #     smiles = self.pdb[0]
    #     assert len(self.pdb) == 1, "This method is only valid for smiles"

    #     import espaloma as esp

    #     # define or load a molecule of interest via the Open Force Field toolkit
    #     from openff.toolkit.topology import Molecule

    #     molecule = Molecule.from_smiles(smiles)

    #     # create an Espaloma Graph object to represent the molecule of interest
    #     molecule_graph = esp.Graph(molecule)

    #     # load pretrained model
    #     espaloma_model = esp.get_model(tag)
    #     espaloma_model = torch.nn.Sequential(
    #         espaloma_model,
    #         esp.mm.geometry.GeometryInGraph(),
    #         esp.mm.energy.EnergyInGraph(),
    #     )

    #     # apply a trained espaloma model to assign parameters
    #     with torch.no_grad():
    #         # go in eval mode:
    #         espaloma_model.eval()
    #         molecule_graph.heterograph.nodes["n1"].data["xyz"] = torch.tensor(self.graph_data["n1"]["xyz"], dtype=torch.float32)
    #         espaloma_model(molecule_graph.heterograph)

    #     # create an OpenMM System for the specified molecule
    #     openmm_system = esp.graphs.deploy.openmm_system_from_graph(molecule_graph, create_system_kwargs={"removeCMMotion":False})


    #     ### get nonbonded energies and gradients from openff:
        

    #     ## DOES ALL NOT WORK BECAUSE OPENFF DOESNT WORK ATM, ENERGIES ARE MUCH TOO HIGH

    #     from openff.toolkit.typing.engines.smirnoff import ForceField as OFFForceField
    #     def load_forcefield(forcefield="openff_unconstrained-2.0.0"):
    #         # get a forcefield
    #         try:
    #             ff = OFFForceField("%s.offxml" % forcefield)
    #         except:
    #             raise NotImplementedError
    #         return ff

    #     ff = load_forcefield()
    #     openmm_system = ff.create_openmm_system(
    #         molecule.to_topology(), charge_from_molecules=[molecule]
    #     )

    #     # delete bonded forces because espaloma get system doesnt work:
    #     delete_force_type = ["HarmonicBondForce", "HarmonicAngleForce", "PeriodicTorsionForce"]
    #     # delete_force_type = ["NonbondedForce"]
    #     if len(delete_force_type) > 0:
    #         i = 0
    #         while(i < openmm_system.getNumForces()):
    #             for force in openmm_system.getForces():
    #                 if any([d.lower() in force.__class__.__name__.lower() for d in delete_force_type]):
    #                     print("Removing force", force.__class__.__name__)
    #                     openmm_system.removeForce(i)
    #                 else:
    #                     i += 1

    #     # get energies and gradients at the locations self.xyz from the openmm system:
    #     from openmm import LangevinMiddleIntegrator
    #     from openmm.app import Simulation
    #     from openmm.unit import Quantity, picosecond, kelvin, kilocalorie_per_mole, picoseconds, angstrom, nanometer, femtoseconds, newton

    #     top = molecule.to_topology().to_openmm()

    #     integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.5*femtoseconds)
    #     simulation = Simulation(topology=top, system=openmm_system, integrator=integrator)

    #     energies, gradients = PDBMolecule.get_data_from_simulation(simulation=simulation, xyz=self.xyz)


    #     # take the bonded energy from the espaloma model energy calculation and the nonbonded energy from openmm:
    #     self.graph_data["g"]["u_esp"] = energies

    #     self.graph_data["n1"]["grad_esp"] = gradients.transpose(1,0,2)

    #     # get the parameters from the graph:
    #     self.graph_data["n2"]["k_esp"] = molecule_graph.heterograph.nodes['n2'].data['k'].numpy()
    #     self.graph_data["n2"]["eq_esp"] = molecule_graph.heterograph.nodes['n2'].data['eq'].numpy()
    #     self.graph_data["n3"]["k_esp"] = molecule_graph.heterograph.nodes['n3'].data['k'].numpy()
    #     self.graph_data["n3"]["eq_esp"] = molecule_graph.heterograph.nodes['n3'].data['eq'].numpy()
    #     self.graph_data["n4"]["k_esp"] = molecule_graph.heterograph.nodes['n4'].data['k'].numpy()

    #     self.graph_data["g"]["u_pred_esp"] = molecule_graph.heterograph.nodes['g'].data['u'].numpy()
    ###########################################################################################