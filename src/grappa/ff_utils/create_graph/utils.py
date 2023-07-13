#%%


import openmm
import openmm.app.topology
from openmm.app import Topology, Element


import numpy as np
import tempfile
import os.path
import torch
import dgl
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import rdchem
from rdkit import Chem
from typing import Union
from pathlib import Path
from typing import List, Tuple, Dict, Union, Callable

from . import read_heterogeneous_graph, read_homogeneous_graph, deploy_parametrize, tuple_indices
from ..charge_models import charge_models
from ... import units as grappa_units
from ...constants import MAX_ELEMENT


def dgl_from_mol(mol:Mol, max_element:int=MAX_ELEMENT)->dgl.DGLGraph:
    """
    The molecule must not contain any high-level chemical information, only the graph structure, atom types and formal charges. Returns a homogeneous graph.
    """
    g = read_homogeneous_graph.from_rdkit_mol(mol, max_element=max_element)
    return g


def bonds_to_rdkit_graph(bond_indices: List[Tuple[int, int]], residues: List[str], atomic_numbers: List[int], atom_names: List[str]=None)->Mol:
    """
    Returns an rdkit molecule for representing the graph structure of the molecule, without chemical details such as bond order, formal charge and stereochemistry.
    Bond indices should be a list of tuples, where each tuple contains the indices of two bonded atoms.
    Residues, atom_names, and atomic_numbers are lists that correspond to each atom in the molecule.
    The indices correspond to the order of the atoms in these lists. The residues must be 3-letter codes, the atom_names PDB-like names and the atomic_numbers the atomic numbers of the atoms.
    If atom_names is None, the formal charge will always be 0. This is a problem if the model has been trained on formal charges as input features.
    Bond indices must be zero-based and with indices corresponding to the order of the atoms in the lists.
    """

    # initialize the molecule
    mol = Chem.RWMol()

    for atomic_number, atom_name, residue in zip(atomic_numbers, atom_names, residues):
        charge = assign_standard_charge(atom_name=atom_name, residue=residue)

        # add the atom to the molecule with the given atomic number and charge
        # atomic number to unsigned int:

        chem_atom = rdchem.Atom(int(atomic_number))
        chem_atom.SetFormalCharge(charge)
        mol.AddAtom(chem_atom)


    # bond_order 1 used for all bonds, regardless what type they are
    for a1, a2 in bond_indices:
        mol.AddBond(a1, a2, rdchem.BondType.SINGLE)

    mol = mol.GetMol()

    return mol




def openmm2rdkit_graph(openmm_top:openmm.app.topology.Topology)->Mol:
    """
    Returns an rdkit molecule for representing the graph structure of the molecule, without chemical details such as bond order, formal charge and stereochemistry.
    To assign formal charges, the topology must contain residues.
    """

    # zero charge used for all atoms, regardless what charge they actually have
    # NOTE: later, find a way to infer charge from pdb, by now we only have standard versions of AAs


    # initialize the molecule
    mol = Chem.RWMol()

    idx_lookup = {} # maps from openmm atom index to openff atom index (usually they agree)
    for i, atom in enumerate(openmm_top.atoms()):
        charge = assign_standard_charge(atom_name=atom.name, residue=atom.residue.name)
        atomic_number = atom.element.atomic_number

        # add the atom to the molecule with the given atomic number and charge
        chem_atom = rdchem.Atom(int(atomic_number))
        chem_atom.SetFormalCharge(charge)
        mol.AddAtom(chem_atom)

        idx_lookup[atom.index] = i

    # bond_order 1 used for all bonds, regardless what type they are
    for bond in openmm_top.bonds():
        mol.AddBond(idx_lookup[bond.atom1.index], idx_lookup[bond.atom2.index], rdchem.BondType.SINGLE)

    mol = mol.GetMol()

    return mol


def assign_standard_charge(atom_name:str, residue:str):
    """
    Assigns standard charges to atoms based on their name and residue.
    """
    if residue in assign_standard_charge.atom_dict.keys():
        if atom_name == assign_standard_charge.atom_dict[residue]:
            return assign_standard_charge.charge_dict[residue]
    # else, the atom is not charged
    return 0
    
assign_standard_charge.atom_dict = {
    "GLU":"OE1",
    "ASP":"OD1",
    "LYS":"NZ",
    "ARG":"NH1",
    "HIS":"ND1", # choose HIE as the standard protonation state
    "HIE":"ND1"
    }

assign_standard_charge.charge_dict = {
    "GLU":-1,
    "ASP":-1,
    "LYS":1,
    "ARG":1,
    "HIS":1,
    "HIE":1
    }


def hard_coded_get_parameters(level:str, g:dgl.DGLGraph, suffix:str="", units:Dict=None)->Dict:
    """
    Returns a dictionary with the parameters and atom indices for the given level as numpy arrays. The permutation symmetry of the n-body term is already divided out, i.e. this is the minimal set of parameters needed to describe the interaction.
    ----------------------
    level: str, the n-body level of the interaction. can be "n1", "n2", "n3", "n4", "n4_improper". e.g. n2 is 2 tuples of 2 atoms, describing a 2-bod- (or bond-) interaction.

    g: dgl.Graph, the graph containing the parameters.

    mol: openff.toolkit.topology.Molecule, the molecule from which the Graph is constructed. This argument will not be needed in a future version.

    suffix: str, the suffix of the parameter names as they are stores in the graph. e.g. if suffix is "_amber99sbildn", will look for something like g.nodes['n1'].data['q_amber99sbildn'] in the graph.
    ----------------------
    This function is not final and will be replaced by a more general function.
    Currently, we calculate the tuple indices using the openff molecule and then check whether the indices are the same as the first ones in the dgl graph since in the dgl graph, the tuples are redundant if permutation symmetry is taken into account.

    For torsion, we assume that only the k parameter is given and the preiodicity is the index in the parameter array + 1.
    For negative k, we assume a phase of pi.
    ----------------------
    Form of the dictionary for n1:
    {
        "idxs":np.array, the indices of the atoms in the molecule that correspond to the parameters. The permutation symmetry of the n-body term is already divided out, i.e. this is the minimal set of parameters needed to describe the interaction.
        "q":np.array, the partial charges of the atoms.
        "sigma":np.array, the sigma parameters of the atoms.
        "epsilon":np.array, the epsilon parameters of the atoms.

        optional (if 'm' or 'mass' in the graph data keys, m has precedence over mass):
            "atom_mass":np.array, the masses of the atoms in atomic units.
    }

    Form of the dictionary for n2/n3:
    {
        "idxs":np.array of shape (#2/3-body-terms, 2/3), the indices of the atoms in the molecule that correspond to the parameters. The permutation symmetry of the n-body term is already divided out, i.e. this is the minimal set of parameters needed to describe the interaction.
        "k":np.array, the force constant of the interaction.
        "eq":np.array, the equilibrium distance of the interaction.   
    }

    Form of the dictionary for n4/n4_improper:
    {
        "idxs":np.array of shape (#4-body-terms, 4), the indices of the atoms in the molecule that correspond to the parameters. The permutation symmetry of the n-body term is already divided out, i.e. this is the minimal set of parameters needed to describe the interaction.
        "ks":np.array of shape (#4-body-terms, n_periodicity), the fourier coefficients for the cos terms of torsion. may be negative instead of the equilibrium dihedral angle (which is always set to zero). n_periodicity is a hyperparemter of the model and defaults to 6.
        "ns":np.array of shape (#4-body-terms, n_periodicity), the periodicities of the cos terms of torsion. n_periodicity is a hyperparemter of the model and defaults to 6.
        "phases":np.array of shape (#4-body-terms, n_periodicity), the phases of the cos terms of torsion. n_periodicity is a hyperparameter of the model and defaults to 6.
    }
    
    """

    lvl = int(level[1])
    if "improper" in level:
        lvl = 5

    # indices of the atom tuples in the graph (i.e. index tuples)
    dgl_indices = g.nodes[level].data["idxs"].detach().numpy()

    
    # if we have atom indices that differ from those in our graph, map back to them:
    if "external_idx" in g.nodes["n1"].data.keys():
        ext_indices = g.nodes["n1"].data["external_idx"].detach().numpy()
        dgl_indices = ext_indices[dgl_indices]

    out_dict = {"idxs":dgl_indices}

    if level == "n1":

        q = g.nodes[level].data[f"q{suffix}"].detach().numpy()

        sigma = g.nodes[level].data[f"sigma{suffix}"].detach().numpy()
        epsilon = g.nodes[level].data[f"epsilon{suffix}"].detach().numpy()

        if not units is None:
            q = openmm.unit.Quantity(q, grappa_units.CHARGE_UNIT).value_in_unit(units["charge"])
            sigma = openmm.unit.Quantity(sigma, grappa_units.DISTANCE_UNIT).value_in_unit(units["distance"])
            epsilon = openmm.unit.Quantity(epsilon, grappa_units.ENERGY_UNIT).value_in_unit(units["energy"])

        out_dict.update({"q": q, "sigma": sigma, "epsilon": epsilon})

        # mass:
        if "m" in g.nodes[level].data.keys():
            mass = g.nodes[level].data[f"m"].detach().numpy()[:, 0]
        elif "mass" in g.nodes[level].data.keys():
            mass = g.nodes[level].data[f"mass"].detach().numpy()[:, 0]

        if not units is None:
            mass = openmm.unit.Quantity(mass, grappa_units.MASS_UNIT).value_in_unit(units["mass"])

        out_dict.update({"mass": mass})


    elif level == "n2":

        k = g.nodes[level].data[f"k{suffix}"].detach().numpy()[:, 0]
        eq = g.nodes[level].data[f"eq{suffix}"].detach().numpy()[:, 0]

        if not units is None:
            k = openmm.unit.Quantity(k, grappa_units.FORCE_CONSTANT_UNIT).value_in_unit(units["energy"]/units["distance"]**2)
            eq = openmm.unit.Quantity(eq, grappa_units.DISTANCE_UNIT).value_in_unit(units["distance"])

        out_dict.update({"k": k, "eq": eq})


    elif level == "n3":

        k = g.nodes[level].data[f"k{suffix}"].detach().numpy()[:, 0]
        eq = g.nodes[level].data[f"eq{suffix}"].detach().numpy()[:, 0]

        if not units is None:
            k = openmm.unit.Quantity(k, grappa_units.ANGLE_FORCE_CONSTANT_UNIT).value_in_unit(units["energy"]/units["angle"]**2)
            eq = openmm.unit.Quantity(eq, grappa_units.ANGLE_UNIT).value_in_unit(units["angle"])

        out_dict.update({"k": k, "eq": eq})

        
    elif level in ["n4", "n4_improper"]:
        ks = g.nodes[level].data[f"k{suffix}"].detach().numpy()
        ns = np.arange(1,ks.shape[1]+1) # one set of periodicities
        ns = np.tile(ns, (ks.shape[0], 1)) # repeat this for all torsions.

        from openmm.unit import Quantity, radians

        pi_in_units = Quantity(np.pi, unit=radians).value_in_unit(grappa_units.ANGLE_UNIT)

        phases = np.where(ks>0, np.zeros_like(ks), pi_in_units*np.ones_like(ks)) # set the phases to pi for negative ks.
        ks = np.abs(ks) # take the absolute value of the ks.

        if not units is None:
            ks = openmm.unit.Quantity(ks, grappa_units.TORSION_FORCE_CONSTANT_UNIT).value_in_unit(units["energy"])

        out_dict.update({
            "ks": ks,
            "ns": ns,
            "phases": phases
            })
        
    return out_dict


def get_improper_positions(g:dgl.DGLGraph)->np.ndarray:
    """
    Returns an array of indices for unique (i.e. with cyclic symmetry divided out) impropers in the graph.
    """
    level = "n4_improper"
    dgl_indices = g.nodes[level].data["idxs"].detach().numpy()

    positions = []
    idx_tuples = []
    for i, idxs in enumerate(dgl_indices):
        # all tuples with cyclic permutations of the non-central atoms:
        invariant_tuples = tuple_indices.get_symmetric_tuples(level="n4_improper", tuple=idxs)
        
        # if there is any tuple that is in the idx_tuples, then we have already seen this improper.
        if any([t in idx_tuples for t in invariant_tuples]):
            continue
        else:
            idx_tuples.append(invariant_tuples[0])
            positions.append(i)
    
    positions = np.array(positions, dtype=np.int32)

    return positions



def get_parameters_from_graph(g, mol, suffix="", units=None):
    """
    Returns a dictionary with the parameters and atom indices for the given level as numpy arrays. The permutation symmetry of the n-body term is already divided out, i.e. this is the minimal set of parameters needed to describe the interaction.
    ----------------------

    g: dgl.Graph, the graph containing the parameters.

    mol: openff.toolkit.topology.Molecule, the molecule from which the Graph is constructed. This argument will not be needed in a future version.

    suffix: str, the suffix of the parameter names as they are stores in the graph. e.g. if suffix is "_amber99sbildn", will look for something like g.nodes['n1'].data['q_amber99sbildn'] in the graph.
    ----------------------
    This function is not final and will be replaced by a more general function.
    Currently, we calculate the tuple indices using the openff molecule and then check whether the indices are the same as the first ones in the dgl graph since in the dgl graph, the tuples are redundant if permutation symmetry is taken into account.

    For torsion, we assume that only the k parameter is given and the preiodicity is the index in the parameter array + 1.
    For negative k, we assume a phase of pi.
    ----------------------
    Form of the dictionary:
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
        "ns":np.array of shape (#4-body-terms, n_periodicity), the periodicities of the cos terms of torsion. n_periodicity is a hyperparemter of the model and defaults to 6.
        "phases":np.array of shape (#4-body-terms, n_periodicity), the phases of the cos terms of torsion. n_periodicity is a hyperparameter of the model and defaults to 6.
    }
    """
    params = {}

    interaction_types = ["atom", "bond", "angle", "proper", "improper"]
    levels = ["n1", "n2", "n3", "n4", "n4_improper"]
    for i_type, lvl in zip(interaction_types, levels):
        p = hard_coded_get_parameters(lvl, g, mol, suffix=suffix, units=units)
        for p_type in p.keys():
            params[f"{i_type}_{p_type}"] = p[p_type]

    return params


def graph_from_pdb(openmm_top, forcefield=openmm.app.ForceField("amber99sbildn.xml"), allow_radicals:bool=True, max_element:int=26, charge_tag:str="amber99sbildn")->dgl.DGLGraph:
    """
    openmm_top: openmm topology
    forcefield: openmm forcefield

    Returns: dgl graph, openff molecule
    In future version, will only return the dgl graph since tuple indices are with symmetry divided out already.
    """
    import openmm.app

    assert isinstance(openmm_top, openmm.app.topology.Topology), f"openmm_top must be an openmm topology, but got {type(openmm_top)}"
    assert isinstance(forcefield, openmm.app.ForceField), f"forcefield must be an openmm forcefield, but got {type(forcefield)}"

    get_charges = charge_models.model_from_dict(tag=charge_tag)

    # get the openff molecule
    mol = openmm2rdkit_graph(openmm_top)

    # get the graph
    g = dgl_from_mol(mol, max_element=max_element)

    # write nonbonded parameters and is_radical in the graph
    g = deploy_parametrize.write_parameters(g=g, topology=openmm_top, forcefield=forcefield, get_charges=get_charges, allow_radicals=allow_radicals)

    # return the graph and the openff molecule
    return g



def graph_from_topology_dict(atoms:List, bonds:List[Tuple[int]], radicals:List[int]=[], max_element:int=26, charge_tag:str="heavy")->dgl.DGLGraph:
    """
    atoms: list of tuples of the form (atom_index, residue, atom_name, (sigma, epsilon), atomic_number)
    bonds: list of tuples of the form (atom_index_1, atom_index_2)
    radicals: list of atom indices that are radicals
    max_element: maximum atomic number for one-hot encoding atom numbers.
    Translates the atom indices to internal indices running from 0 to num_atoms-1. Then stores the external indices in the graph such that they can be recovered later.
    atom_idx = g.nodes["n1"].data["external_idx"][internal_idx] gives the external index of the atom with internal index internal_idx.
    """
    atom_types = [a_entry[1] for a_entry in atoms]
    residues = [a_entry[2] for a_entry in atoms]
    sigmas = [a_entry[4][0] for a_entry in atoms]
    epsilons = [a_entry[4][1] for a_entry in atoms]
    atomic_numbers = [a_entry[5] for a_entry in atoms]

    # residue indices are only needed if there are radicals
    residue_indices = None
    if len(radicals) > 0:
        residue_indices = [a_entry[3] for a_entry in atoms]

    external_idxs = [a_entry[0] for a_entry in atoms] # i-th entry is the index of the i-th atom in the molecule

    external_to_internal_idx = {external_idxs[i]:i for i in range(len(external_idxs))} # i-th entry is the list-position of the atom with index i

    bonds = [(external_to_internal_idx[bond[0]], external_to_internal_idx[bond[1]]) for bond in bonds]

    radical_indices = [external_to_internal_idx[radical] for radical in radicals]

    # calculate charges
    charge_model = charge_models.model_for_atoms(tag=charge_tag)

    charges = charge_model(atom_names=atom_types, residues=residues, rad_indices=radical_indices, residue_indices=residue_indices)


    mol = bonds_to_rdkit_graph(bond_indices=bonds, residues=residues, atomic_numbers=atomic_numbers, atom_names=atom_types)

    # create the graph
    g = dgl_from_mol(mol=mol, max_element=max_element)


    # write nonbonded parameters, charges and is_radical in the graph
    g.nodes["n1"].data["is_radical"] = torch.zeros(len(atomic_numbers), 1, dtype=torch.float32)
    if not radicals is None:
        g.nodes["n1"].data["is_radical"][radical_indices] = 1.
    
    # one-hot encode the residue:
    from ...units import RESIDUES
    g.nodes["n1"].data["residue"] = torch.zeros(len(atomic_numbers), len(RESIDUES), dtype=torch.float32)
    for idx, resname in zip(range(len(residues)), residues):
        if resname in RESIDUES:
            res_index = RESIDUES.index(resname) # is unique
            g.nodes["n1"].data["residue"][idx] = torch.nn.functional.one_hot(torch.tensor((res_index)).long(), num_classes=len(RESIDUES)).float()
        else:
            raise ValueError(f"Residue {resname} not in {RESIDUES}")


    g.nodes["n1"].data["q"] = torch.tensor(charges, dtype=torch.float32).unsqueeze(dim=1)
    g.nodes["n1"].data["q_ref"] = torch.tensor(charges, dtype=torch.float32).unsqueeze(dim=1)
    g.nodes["n1"].data["sigma"] = torch.tensor(sigmas, dtype=torch.float32).unsqueeze(dim=1)
    g.nodes["n1"].data["epsilon"] = torch.tensor(epsilons, dtype=torch.float32).unsqueeze(dim=1)

    g.nodes["n1"].data["external_idx"] = torch.tensor(external_idxs, dtype=torch.int64).unsqueeze(dim=1)
    return g


def process_input(input_, classical_ff=openmm.app.ForceField("amber99sbildn.xml"))->dgl.DGLGraph:

    import openmm.app.topology
    from openmm.app import PDBFile
    if type(input_) == openmm.app.topology.Topology:
        return graph_from_pdb(input_, forcefield=classical_ff)
    elif type(input_) == PDBFile:
        return graph_from_pdb(input_.topology, forcefield=classical_ff)
    elif type(input_) == str:
        return graph_from_pdb(PDBFile(input_).topology, forcefield=classical_ff)
    
    elif type(input_) == dict:
        assert "atoms" in input_.keys(), "input_ dictionary must contain an 'atoms' key"
        assert "bonds" in input_.keys(), "input_ dictionary must contain a 'bonds' key"

        radicals=input_["radicals"] if "radicals" in input_.keys() else []

        return graph_from_topology_dict(atoms=input_["atoms"], bonds=input_["bonds"], radicals=radicals)
    else:
        raise TypeError(f"input_ must be either a string, a PDBFile or a Topology, but got {type(input_)}")
    

def process_output(g, input_type, classical_ff:openmm.app.ForceField=openmm.app.ForceField("amber99sbildn.xml"), topology=None, system_kwargs:Dict=None, units:Dict=None)->Union[openmm.System, Dict]:
    """
    If the input type is an openmm topology or an openmm PDBFile the output is an openmm system.
    If the input is a path to a PDB file, the output is a dictionary containing indices describing the interactions. If the input is a path to a gromacs topology file, the output is a topology file with parameters added. This has to be made evident by the file suffix (.gro or .pdb).
    """

    # RETURN OPENMM SYSTEM
    if input_type in [openmm.app.topology.Topology, openmm.app.PDBFile]:
        assert not topology is None, "topology must be given if input_type is openmm.app.topology.Topology or openmm.app.PDBFile"

        # calculate the param dict with default grappa units since they are converted when creating the openmm system 
        param_dict = get_parameters_from_graph(g=g, units=None)

        openmm_system = deploy_parametrize.openmm_system_from_params(param_dict=param_dict, topology=topology, classical_ff=classical_ff, allow_radicals=True, system_kwargs=system_kwargs)

        return openmm_system


    # RETURN PARAMETER DICT
    elif input_type in [str, dict]:
        param_dict = get_parameters_from_graph(g=g, units=units)

        return param_dict

    else:
        raise TypeError(f"invalid type argument: {input_type}")
    
    


def bonds_to_openmm(bonds:List[Tuple[int, int]], residue_indices:List[int], residues:List[str], atom_types:List[str], atomic_numbers:List[int], ordered_by_res:bool=True)->Topology:

    # create a new, empty topology
    openmm_topology = Topology()

    # create a new chain (assuming all residues are in the same chain)
    chain = openmm_topology.addChain()

    if ordered_by_res:
        # create a new residue every time the residue index changes:
        last_res_index = None
    else:
        # store all residue indices:
        added_res = []

    for atom_idx, (res_index, res, atom_type, atomic_number) in enumerate(zip(residue_indices, residues, atom_types, atomic_numbers)):

        if ordered_by_res:
            # check if we need to start a new residue
            if res_index != last_res_index:
                residue = openmm_topology.addResidue(res, chain)
                last_res_index = res_index
        else:
            if res_index not in added_res:
                residue = openmm_topology.addResidue(res, chain)
                added_res.append(res_index)

        # determine element based on atom type
        # this is just a basic example; you may need to map atom types to elements differently
        element = Element.getByAtomicNumber(atomic_number)

        # add the atom to the current residue
        openmm_topology.addAtom(name=atom_type, element=element, residue=residue, id=atom_idx)

    # add the bonds to the topology:
    atom_list = list(openmm_topology.atoms())

    # Iterate over bond_list and add each bond to the topology
    for bond in bonds:
        atom1 = atom_list[bond[0]]
        atom2 = atom_list[bond[1]]
        openmm_topology.addBond(atom1, atom2)
        
    return openmm_topology

