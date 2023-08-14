#%%
import openmm
from openmm import unit
from openmm.app import topology

from openmm.unit import Quantity
import json
import numpy as np
from typing import List, Tuple, Dict, Union, Callable
from pathlib import Path
from ..create_graph.find_radical import get_radicals
from ..create_graph.read_pdb import replace_h23_to_h12, one_atom_replace_h23_to_h12
from ..classical_ff.collagen_utility import get_mod_amber99sbildn
from ... import units as grappa_units

CHARGE_DICT_UNIT = unit.elementary_charge
#%%

def get_path_from_tag(tag:str):
    basepath = Path(__file__).parent / Path("charge_dicts")
    if not tag is None:
        if tag == "bmk":
            d_path = basepath / Path("BMK/AAs_nat.json")
            drad_path = basepath / Path("BMK/AAs_rad.json")
        elif tag == "avg":
            d_path = basepath / Path("ref/AAs_nat.json")
            drad_path = basepath / Path("ref/AAs_rad_avg.json")
        elif tag == "heavy":
            d_path = basepath / Path("ref/AAs_nat.json")
            drad_path = basepath / Path("ref/AAs_rad_heavy.json")
        elif tag == "amber99sbildn":
            d_path = basepath / Path("ref/AAs_nat.json")
            drad_path = basepath / Path("ref/AAs_rad_heavy.json")
        else:
            raise ValueError(f"tag {tag} not recognized")
    return d_path, drad_path



def model_from_dict(tag:str=None, d_path:Union[str, Path]=None, d_rad_path:Union[str, Path]=None):
    """
    Returns a function that takes a topology and returns a list of charges for each atom in the topology. Uses the dictionary stored in the json file a d_path.
    possible tags: ['bmk', 'avg', 'heavy, 'amber99sbildn']
    By default, uses heavy for radicals if the tag is amber99sbildn.
    random: randomly assigns charges, keeping the total charge obtained with the tagged model invariant
    """
    if d_path is None and tag is None:
        raise ValueError("Either d_path or tag must be given")
    
    if d_path is None and d_rad_path is None:
        d_path, d_rad_path = get_path_from_tag(tag)

    if d_path is None and not d_rad_path is None:
        d_path = get_path_from_tag(tag)[0]

    with open(str(d_path), "r") as f:
        d = json.load(f)

    with open(str(d_rad_path), "r") as f:
        d_rad = json.load(f)
    
    # add ACE and NME from ref (this corresponds to amber99sbildn) if not present
    if (not "ACE" in d.keys()) or (not "NME" in d.keys()):
        with open(str(Path(__file__).parent / Path("charge_dicts/ref/AAs_nat.json")), "r") as f:
            ref = json.load(f)
        for key in ["ACE", "NME"]:
            if not key in d.keys():
                d[key] = ref[key]

    def get_charges(top:topology, radical_indices:List[int]=[]):
        return from_dict(d=d, top=top, d_rad=d_rad, radical_indices=radical_indices)
    
    return get_charges

def model_for_atoms(tag:str=None, d_path:Union[str, Path]=None, d_rad_path:Union[str, Path]=None):
    """
    Returns a function that takes:
        
        atom_names: list of atom names
        residues: list of residue names
        d: dictionary of charges, keys are residue names, values are dictionaries of atom names and charges
        d_rad: dictionary of charges for radicals, keys are residue names, values are dictionaries of radical names and charges
        rad_indices: list of indices of atoms that are radicals
        residue_indices: list of the residue indices for each atom

        Radical indices must be provided if radicals are present, otherwise the charges will not add up to integer values! If radical indices are provided, the radical dictionary and a list of residue_indices must also be provided.
        Residue_indices must be provided if radicals are present and is a list of length len(atom_names) that maps each atom to its residue index as in the pdb file.

    and returns a list of charges for each atom in the topology. Uses the dictionary stored in the json file a d_path.
    possible tags: ['bmk', 'avg', 'heavy, 'amber99sbildn']
    By default, uses heavy for radicals if the tag is amber99sbildn.
    random: randomly assigns charges, keeping the total charge obtained with the tagged model invariant
    """
    if d_path is None and tag is None:
        raise ValueError("Either d_path or tag must be given")
    
    if d_path is None and d_rad_path is None:
        d_path, d_rad_path = get_path_from_tag(tag)

    if d_path is None and not d_rad_path is None:
        d_path = get_path_from_tag(tag)[0]

    with open(str(d_path), "r") as f:
        d = json.load(f)

    with open(str(d_rad_path), "r") as f:
        d_rad = json.load(f)
    
    # add ACE and NME from ref (this corresponds to amber99sbildn) if not present
    if (not "ACE" in d.keys()) or (not "NME" in d.keys()):
        with open(str(Path(__file__).parent / Path("charge_dicts/ref/AAs_nat.json")), "r") as f:
            ref = json.load(f)
        for key in ["ACE", "NME"]:
            if not key in d.keys():
                d[key] = ref[key]

    def get_charges(atom_names:List[str], residues:List[str], rad_indices:List[int]=[], residue_indices:List[int]=None):
        return from_atoms_dict(d=d, d_rad=d_rad, rad_indices=rad_indices, residue_indices=residue_indices, atom_names=atom_names, residues=residues)
    
    return get_charges


def randomize_model(get_charges, noise_level:float=0.1):
    def get_random_charges(top, radical_indices=[]):
        charges = get_charges(top, radical_indices=radical_indices)
        unit = charges[0].unit
        charges = np.array([q.value_in_unit(unit) for q in charges])
        charge = np.sum(charges)
        charges = np.array(charges)

        noise = np.random.normal(0, noise_level, size=charges.shape)
        # create noise with zero mean, i.e. the total charge is conserved
        noise -= np.mean(noise)
        charges += noise

        charges = [openmm.unit.Quantity(q, unit) for q in charges]
        return charges
    return get_random_charges


def rename_caps(name:str, res:str):
    """
    hard code the H naming of caps (maybe do this in the to_openmm function instead?)
    also hard-code the naming of the terminal atoms.
    """
    if res == "ACE":
        if "H" in name and not "C" in name:
            name = "HH31"
        if name == "CT":
            name = "CH3"
    elif res == "NME":
        if "H" in name and not "C" in name:
            if len(name) > 1:
                name = "HH31"
        if "C" in name:
            name = "CH3"
    return name


def from_dict(d:dict, top:topology, d_rad:dict=None, radical_indices:List[int]=[]):
    """
    Radical indices is for obtaining the radical indices without returning them. We use a List because they are passed like a C++ style reference.
    """

    charge_unit=grappa_units.CHARGE_UNIT

    charges = []
    # replace _radical_indices in-place

    radical_indices[:], rad_names, res_radical_indices = get_radicals(topology=top, forcefield=get_mod_amber99sbildn())

    assert len(set(res_radical_indices)) == len(res_radical_indices), f"radical indices are not unique: {res_radical_indices}. this might be due to a residue in which two radicals occur."

    if len(res_radical_indices) != 0 and d_rad is None:
        raise ValueError("radical indices are given but no radical dictionary is given")


    for atom in top.atoms():
        rad_name = None
        res_idx = atom.residue.index

        num_atoms = len([a for a in atom.residue.atoms()])

        if res_idx in res_radical_indices:
            rad_name = rad_names[res_radical_indices.index(res_idx)]



        # IF ID IN RADICAL INDICES, DICT IS GIVEN BY THE RAD NAME
        res = atom.residue.name
        name = atom.name


        name = one_atom_replace_h23_to_h12(name, res)

        #############################################################
        # for isoleucine, identify the correct CG atom. in our case, CG1 is the one with two hydrogens, i.e. the one bonded to two Cs
        if res == "ILE":
            renaming_needed = None
            residue = atom.residue
            # find the atom named CG1.
            # determine the number of bonds to C atoms
            # if this number is two, renaming_needed is False, if it is 1: True, else exception.
        
            cg1_atom = None
            for atom in residue.atoms():
                if atom.name == "CG1":
                    cg1_atom = atom
                    break

            if cg1_atom is None:
                raise RuntimeError("CG1 atom not found in ILE residue")

            # Count the number of carbon atoms bonded to CG1
            carbon_count = 0
            for bond in residue.bonds():
                if bond[0] == cg1_atom or bond[1] == cg1_atom:
                    if bond[0].element.symbol == "C" and bond[1].element.symbol == "C":
                        carbon_count += 1

            if carbon_count == 2:
                renaming_needed = False
            elif carbon_count == 1:
                renaming_needed = True
            else:
                raise RuntimeError(f"Unexpected number of carbon atoms bonded to CG1 in ILE residue: {carbon_count}")



            # then, if renaming_needed is True, replace name if it is either in the keys or values of the dict:
            replacements = {
                "CG1":"CG2",
                "HG11":"HG21",
                "HG12":"HG22",
                "HG13":"HG23",
            }
            if renaming_needed:
                if name in replacements.keys():
                    name = replacements[name]
                elif name in replacements.values():
                    name = [k for k,v in replacements.items() if v == name][0]
        #############################################################

        # hard code that we only have HIE atm
        if res == "HIS":
            res = "HIE"


        lvls = ["A", "B", "G", "D", "E", "Z", "H"]

        # check whether res is in the dict:
        if not res in (d.keys() if rad_name is None else d_rad.keys()):
            raise ValueError(f"Residue {res} not in dictionary, residues are {d.keys() if rad_name is None else d_rad.keys()}")
        
        # check whether rad_name is in the dict in the radical case:
        if not rad_name is None:
            if not rad_name in d_rad[res].keys():
                raise ValueError(f"Radical {rad_name} not in dictionary for residue {res}, radicals are {d_rad[res].keys()}")

        # pick the dictionary to use:
        d_used = d[res] if rad_name is None else d_rad[res][rad_name]


        if not rad_name is None:
            # rename all HB2 to HB1 because these are in the dict for radicals and have the same parameters since they are indisinguishable
            if not name in d_used.keys():
                orig_name = name
                if name in [f"H{lvl}2" for lvl in lvls]:
                    name = name.replace("2", "1")
                elif name in [f"H{lvl}3" for lvl in lvls]:
                    name = name.replace("3", "1")
            if not name in d_used.keys():
                raise ValueError(f"{name} not in radical dictionary for {res} {rad_name}.\nStored atom names are {d_used.keys()}")



        # hard code the H naming of caps (maybe do this in the to_openmm function instead?)
        name = rename_caps(name=name, res=res)
        

        if name not in d_used:
            # hard code some exceptions that occur sometimes. replace first entry by second.
            replacements = {
                "ILE":("CD1", "CD"),
                }
            if res in replacements.keys():
                if name == replacements[res][0]:
                    name = replacements[res][1]

            # if it is still not in the dict, raise an error:
            if name not in d_used:
                raise ValueError(f"(Atom {name}, Residue {res}, Radical {rad_name}) not in the corresponding dictionary,\nnames in dict are {d_used.keys()}")

        charge = float(d_used[name])

        if len(d_used.keys()) != num_atoms:
            raise RuntimeError(f"Residue {res} has {num_atoms} atoms, but dictionary has {len(d_used.keys())} atoms.\ndictionary entries: {d_used.keys()},\natom names: {[a.name for a in atom.residue.atoms()]},\nrad name is {rad_name},\nres_idx is {res_idx}, \nres_radical_indices are {res_radical_indices},\nrad_names are {rad_names}")

        charge = openmm.unit.Quantity(charge, CHARGE_DICT_UNIT).value_in_unit(charge_unit)
        charges.append(charge)

    return charges



def get_rad_names(atom_names:List[str], rad_indices:List[int]=[], residue_indices:List[int]=None):
    """
    Not used anymore.
    """

    rad_names = np.array([None]*len(atom_names))

    if len(rad_indices) > 0:
        # Convert lists to numpy arrays
        rad_indices = np.array(rad_indices)
        residue_indices = np.array(residue_indices)

        # Get unique residues that contain a radical
        unique_rad_residues = np.unique(residue_indices[rad_indices])

        assert len(unique_rad_residues) == len(rad_indices), f"Each residue should contain exactly one radical, but {len(unique_rad_residues)} residues contains {len(rad_indices)} radicals."

        # map res index to radical name:
        idx_to_name = {residue_indices[rad_idx]:atom_names[rad_idx] for rad_idx in rad_indices}

        # Loop over unique residues
        for residue_idx in unique_rad_residues:
            # Get boolean mask for atoms in the same residue
            atom_mask = residue_indices == residue_idx

            # Get the atom name of the radical
            rad_name = idx_to_name[residue_idx]

            # Assign the name of the radical to all atoms in this residue
            rad_names[atom_mask] = rad_name

    return rad_names.tolist()



def from_atoms_dict(d:dict, atom_names:List[str], residues:List[str], d_rad:dict=None, rad_indices:List[int]=[], residue_indices:List[int]=None):
    """
    atom_names: list of atom names
    residues: list of residue names
    d: dictionary of charges, keys are residue names, values are dictionaries of atom names and charges
    d_rad: dictionary of charges for radicals, keys are residue names, values are dictionaries of radical names and charges
    rad_indices: list of indices of atoms that are radicals
    residue_indices: list of the residue indices for each atom

    Radical indices must be provided if radicals are present, otherwise the charges will not add up to integer values! If radical indices are provided, the radical dictionary and a list of residue_indices must also be provided.
    Residue_indices must be provided if radicals are present and is a list of length len(atom_names) that maps each atom to its residue index as in the pdb file.
    """

    charge_unit=grappa_units.CHARGE_UNIT

    if len(rad_indices) != 0 and d_rad is None:
        raise ValueError("radical indices are given but no radical dictionary is given")
    if len(rad_indices) != 0 and residue_indices is None:
        raise ValueError("radical indices are given but no residue indices are given")

    charges = []

    rad_names = get_rad_names(atom_names, rad_indices, residue_indices)

    # only used for error checking:
    num_atoms_per_res = None
    if not residue_indices is None:
        num_atoms_per_res = np.bincount(residue_indices)
    
    for idx_, name, res, rad_name in zip(range(len(atom_names)), atom_names, residues, rad_names):

        # IF ID IN RADICAL INDICES, DICT IS GIVEN BY THE RAD NAME
        # exactly as in the function above usnig the openmm topology now, we are in a residue with radical if rad_name is not None.

        name = one_atom_replace_h23_to_h12(name, res)

        # hard code that we only have HIE atm
        if res == "HIS":
            res = "HIE"

        H23_dict = replace_h23_to_h12.d

        if not rad_name is None:
            # rename all HB2 to HB1 because these are in the dict for radicals and have the same parameters since they are indisinguishable
            if not res in d_rad.keys():
                raise ValueError(f"Residue {res} not in radical dictionary, residues are {d_rad.keys()}")
            if not name in d_rad[res][rad_name].keys():
                if res in H23_dict.keys():
                    orig_name = name
                    if name in [f"H{lvl}2" for lvl in H23_dict[res]]:
                        name = name.replace("2", "1")
                    elif name in [f"H{lvl}1" for lvl in H23_dict[res]]:
                        name = name.replace("1", "2")
            if not name in d_rad[res][rad_name].keys():
                raise ValueError(f"Neither {name} nor {orig_name} in radical dictionary for {res} {rad_name}.\nStored atom names are {d_rad[res][rad_name].keys()}")

        name = rename_caps(name=name, res=res)

        if not res in (d.keys() if rad_name is None else d_rad.keys()):
            raise ValueError(f"Residue {res} not in dictionary, residues are {d.keys() if rad_name is None else d_rad.keys()}")
            
        if not rad_name is None and not rad_name in d_rad[res].keys():
            raise ValueError(f"Radical {rad_name} not in dictionary for residue {res}, radicals are {d_rad[res].keys()}")

        if name not in (d[res] if rad_name is None else d_rad[res][rad_name]):
            raise ValueError(f"(Atom {name}, Residue {res}, Radical {rad_name}) not in dictionary, atom names are {d[res].keys() if rad_name is None else d_rad[res][rad_name].keys()}")

        charge = float(d[res][name]) if rad_name is None else float(d_rad[res][rad_name][name])

        if not num_atoms_per_res is None:
            if len((d[res] if rad_name is None else d_rad[res][rad_name]).keys()) != num_atoms_per_res[residue_indices[idx_]]:
                raise RuntimeError(f"Residue {res} with index {residue_indices[idx_]} has {num_atoms_per_res[residue_indices[idx_]]} atoms, but dictionary has {len((d[res] if rad_name is None else d_rad[res][rad_name]).keys())} atoms.\ndictionary entries: {(d[res] if rad_name is None else d_rad[res][rad_name]).keys()}")

        charge = openmm.unit.Quantity(charge, CHARGE_DICT_UNIT).value_in_unit(charge_unit)
        charges.append(charge)

    return charges


#%%
if __name__=="__main__":
    from ...PDBData.PDBDataset import PDBDataset
    get_charges_bmk = model_from_dict("bmk")
    spicepath = Path(__file__).parent.parent.parent.parent / Path("mains/small_spice")
    dspath = Path(spicepath)/Path("small_spice.hdf5")
    ds = PDBDataset.from_spice(dspath, n_max=1)
    # %%
    ds.filter_validity()
    len(ds)

    ds.filter_confs()
    len(ds)
    #%%
    for mol in ds.mols:
        top = mol.to_openmm().topology
        charges = get_charges_bmk(top)
        print(charges)
    #%%
    ds.parametrize(get_charges=get_charges_bmk, suffix="_bmk", charge_suffix="_bmk")
    #%%
    i = 0
    g = ds[i].to_dgl()
    print(g.nodes["n1"].data["q_bmk"])
    # %%
    import copy
    ds2 = copy.deepcopy(ds)
    ds2.parametrize()
    #%%
    g = ds2[i].to_dgl()
    print(g.nodes["n1"].data["q_ref"])
    #%%
    ds3 = copy.deepcopy(ds)
    ds3.parametrize(get_charges=model_from_dict("amber99sbildn"))
    g = ds3[i].to_dgl()
    print(g.nodes["n1"].data["q_ref"])
    #%%
    #############################
    # TEST RADICALS
    from ...PDBData.PDBMolecule import PDBMolecule
    rad = PDBMolecule.from_pdb("./../../../mains/tests/radicals/F_radA.pdb")
    mol = PDBMolecule.from_pdb("./../../../mains/tests/radicals/F.pdb")
    fail_rad = PDBMolecule.from_pdb("./../../../mains/tests/radicals/F_rad.pdb")
    # print(*mol.pdb)
    print(*rad.pdb)
    radical_indices = []
    charges = get_charges_bmk(rad.to_openmm().topology, radical_indices=radical_indices)
    # print(charges)
    ds = PDBDataset()
    ds.mols = [rad, mol]
    # rad indices gets modified in-place
    radical_indices
    #%%

    #%%
    ds.parametrize(get_charges=get_charges_bmk, allow_radicals=True, suffix="_bmk", charge_suffix="_bmk")
    #%%
    #%%
    g_rad = ds[0].to_dgl()
    g_mol = ds[1].to_dgl()
    print(g_rad.nodes["n1"].data["q_bmk"])
    print(g_mol.nodes["n1"].data["q_bmk"])
    #%%
    print(g_rad.nodes["n1"].data["is_radical"])
    #%%
    # print both charges next to each other:
    mol_atoms = [a for a in mol.to_openmm().topology.atoms()]
    rad_atoms = [a for a in rad.to_openmm().topology.atoms()]

    print("radical charge  molecule charge  radical atom  molecule atom")

    i = 0
    j = 0
    while i < len(mol_atoms):
            
        if mol_atoms[i].name != rad_atoms[j].name:
            print(f"   ---   {g_mol.nodes['n1'].data['q_bmk'][i].item():>3.3f}   - {mol_atoms[i].name}".replace(" -", "-"))
            i += 1
        else:
            print(f" {g_rad.nodes['n1'].data['q_bmk'][j].item():>3.3f}  {g_mol.nodes['n1'].data['q_bmk'][i].item():>3.3f}  {rad_atoms[j].name} {mol_atoms[i].name}".replace(" -", "-"))
            i += 1
            j += 1


    # %%

# %%

# %%
