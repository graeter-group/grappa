
from openmm.app import PDBFile

from typing import List, Tuple, Dict, Union, Callable

def replace_h23_to_h12(pdb:PDBFile):
    """
    for every residue, remap Hs (2,3) -> (2,1), eg HB2, HB3 -> HB2, HB1.
    """

    for atom in pdb.topology.atoms():
        if atom.residue.name in replace_h23_to_h12.d.keys():
            for level in replace_h23_to_h12.d[atom.residue.name]:
                for i in ["3"]:
                    if atom.name == f"H{level}{i}":
                        atom.name = f"H{level}{int(i)-2}"
                        continue
        elif atom.residue.name == "ILE":
            for i in ["3"]:
                if atom.name == f"HG1{i}":
                    atom.name = f"HG1{int(i)-2}"
                    continue
            for i in ["1","2","3"]:
                if atom.name == f"HD1{i}":
                    atom.name = f"HD{i}"
            if atom.name == f"CD1":
                atom.name = f"CD"
                continue

    return pdb

replace_h23_to_h12.d = {"CYS":["B"], "ASP":["B"], "GLU":["B","G"], "PHE":["B"], "GLY":["A"], "HIS":["B"], "HIE":["B"], "LYS":["B","G","D","E"], "LEU":["B"], "MET":["B","G"], "ASN":["B"], "PRO":["B","G","D"], "GLN":["B", "G"], "ARG":["B","G","D"], "SER":["B"], "TRP":["B"], "TYR":["B"]}


def one_atom_replace_h23_to_h12(name:str, resname:str):
    """
    for one atom, apply the mapping of Hs (2,3) -> (2,3), eg HB2, HB3 -> HB2, HB1.
    """

    if resname in replace_h23_to_h12.d.keys():
        for level in replace_h23_to_h12.d[resname]:
            name = name.replace(f"H{level}{3}", f"H{level}{1}")
    return name
    


def in_list_replace_h23_to_h12(names:List[str], resname:str):
    """
    remap Hs (2,3) -> (2,3), eg HB2, HB3 -> HB2, HB1.
    """

    for i,_ in enumerate(names):
        names = one_atom_replace_h23_to_h12(names[i], resname)
    return names




def replace_h12_to_h23(topology):
    """
    simply inverts the first part of replace_h23_to_h12.
    """
    for atom in topology.atoms():
        if atom.residue.name in replace_h23_to_h12.d.keys():
            for level in replace_h23_to_h12.d[atom.residue.name]:
                    if atom.name == f"H{level}1":
                        atom.name = f"H{level}3"
                        continue
    return topology