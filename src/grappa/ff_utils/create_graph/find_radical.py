#%%
from .read_pdb import one_atom_replace_h23_to_h12
import openmm
from openmm import unit
from openmm.app import Simulation
from openmm.app import ForceField, topology

from openmm.unit import Quantity
import json
import numpy as np
from typing import List, Tuple, Dict, Union, Callable
from pathlib import Path
import copy

#%%

def add_radical_residues(forcefield, topology):
    """
    For each residue that is not matched to a template, add a template with the same atoms but with the Hs removed to be able to parametrize radicals.
    NOTE: This might cause issues for histidin if the histidin type cannot be determined due to the missing Hs.
    """

    forcefield = copy.deepcopy(forcefield)

    # effectively delete the residues that can be mistaken with radicals:
    # LYN and CYM
    ######################
    try:
        matches = forcefield.getMatchingTemplates(topology)
    except ValueError:
        # this happens when there are no matched residues
        matches = None
    
    if not matches is None:
        for (residue, match) in zip(topology.residues(), matches):
            if match.name in generate_unmatched_templates.corrections.keys():
                match.name = ""
    ######################

    [templates, residues] = generate_unmatched_templates(topology=topology, forcefield=forcefield)

    for t_idx, template in enumerate(templates):
        resname = template.name
        if resname == "HIS":
            resname = "HIE"
        ref_template = forcefield._templates[resname]
        # the atom names stored in the template of the residue
        ref_names = [a.name for a in ref_template.atoms]
        ref_names = [one_atom_replace_h23_to_h12(n, resname=resname) for n in ref_names]

        # check whether all atoms can be matched:
        atom_names = [a.name for a in template.atoms]
        diff1 = set(atom_names) - set(ref_names)
        diff2 = set(ref_names) - set(atom_names)
        if len(diff2) > 2 or len(diff1) > 0:
            raise ValueError(f"Template {template.name} does not match reference template:\nIn pdb, not in reference: {diff1}\nIn reference, not in pdb:{diff2},\nallowed is at most one Hydrogen atom that is not in the pdb and no atom that is not in the reference.")
            

        for atom in template.atoms:
            name = atom.name

            # find the atom with that name in the reference template
            try:
                ref_idx = ref_names.index(name)
            except ValueError:
                print(f"Atom {name} not found in reference template {template.name}: {ref_names}")
                raise
            atom.type = ref_template.atoms[ref_idx].type

        # create a new template
        template.name = template.name +f"_rad_{t_idx}"
        forcefield.registerResidueTemplate(template)

    for res in generate_unmatched_templates.corrections.keys():
        forcefield._templates.pop(res)

    return forcefield


def generate_unmatched_templates(topology:topology, forcefield:ForceField=ForceField("amber99sbildn.xml")):
    """
    Removes the entries LYN and CYM from the forcefield.
    This function extends generateTemplatesForUnmatchedResidues from openmm:
    In the forcefield there are some templated that match the radical, i.e. the uncharged version of a positively charged residue. we add these to the templates to the unmatched ones with the corrected name. we can do this because our ff only applies to the standard versions of amino acids, any deviation is assumed to be caused by being a radical.
    Not guaranteed to work properly for forcefields other than amber99sbildn.
    """
    [templates, residues] = forcefield.generateTemplatesForUnmatchedResidues(topology)
    
    corrections = generate_unmatched_templates.corrections

    try:
        matches = copy.deepcopy(forcefield.getMatchingTemplates(topology))
    except ValueError:
        # this happens when there are no matched residues
        matches = None
    
    if not matches is None:
        for (residue, match) in zip(topology.residues(), matches):
            if match.name in corrections.keys():
                original_res = corrections[match.name]
                # fake a generated fail-template:
                match.name = original_res
                for i,_ in enumerate(match.atoms):
                    # rename some Hs according to the PDBMolecule convention:
                    match.atoms[i].name = one_atom_replace_h23_to_h12(match.atoms[i].name, resname=original_res)
                    
                    # delete the types so that they are regenerated later:
                    match.atoms[i].type = None

                templates.append(match)
                residues.append(residue)

    return templates, residues


# these are empirical for the amber99sbildn forcefield
generate_unmatched_templates.corrections = {"LYN":"LYS", "CYM":"CYS"}


def get_radicals(topology:topology, forcefield:ForceField=ForceField("amber99sbildn.xml")):
    """
    Returns the indices of the residues in which radicals occur and  as radical_indices, radical_names, residue_indices: List[int], List[str], List[int]

    Find the radicals in a topology. Do this using the generate_missing... method from openmm forcefields. then collect all atoms names to find the missing name. then figure out to which atom this H is connected. this way we might not be able to differentiate between charge and radical, but for standard cases it should work.
    NOTE: currently only works for one radical per residue, i.e. one missing hydrogen per residue.
    """
    # get the missing hydrogens assuming that the ff cannot assign residues containing a radical
    
    residue_indices = []
    radical_names = []
    radical_indices = []

    [templates, residues] = generate_unmatched_templates(topology=topology, forcefield=forcefield)

    for t_idx, template in enumerate(templates):
        resname = template.name
        if resname == "HIS":
            resname = "HIE"

        ref_template = forcefield._templates[resname]
        # the atom names stored in the template of the residue
        ref_names = [a.name for a in ref_template.atoms]
        ref_names = [one_atom_replace_h23_to_h12(n, resname=resname) for n in ref_names]

        if len(template.atoms) != len(ref_template.atoms)-1:
            raise ValueError(f"Encountered {resname} with {len(template.atoms)} atoms that cannot be matched to reference {len(ref_template.atoms)}. The reason for this cannot be a single missing hydrogen.")

        # delete the H indices since the Hs are not distinguishable anyways
        # in the template, the Hs are named HB2 and HB3 while in some pdb files they are named HB1 and HB2



        # find the atom name of the one that is not in the template
        names = [a.name for a in template.atoms]

        assert len(ref_names) == len(set(ref_names)), f"Found duplicate atom names in residue {resname}."

        diff = []
        for n in ref_names:
            try:
                # search the names for the atom. we already know that names has less elements than ref_names
                # if it is found, remove it out of the names list.
                # this is necessary because we can have two HB atoms in the residue, but only one in the radical, i.e. number of occurence is important

                idx = names.index(n)
                names.pop(idx)
            except ValueError:
                diff.append(n)
        if len(diff) != 1:
            errstr = f"Could not find the missing atom in {resname}.\nUnmatched Atom names are          {names}\nand reference names are {ref_names},\ndiff is {diff}."
            raise ValueError(errstr)
        
        # if the missing atom is a hydrogen, its neighbor is a radical.
        # thus, loop over all atoms in the residue and find the one that is connected to the missing atom
        for atom in ref_template.atoms:
            ref_name = atom.name
            ref_name = one_atom_replace_h23_to_h12(name=ref_name, resname=resname)
            if ref_name == diff[0]:
                if len(atom.bondedTo)!=1:
                    raise ValueError(f"Found {len(atom.bonds)} bonds for atom {ref_name} in residue {resname}, but expected 1 since we expect to find missing hydrogens (that would be bonded to exactly one atom).") 
                # append the atom name to the radical sequence
                radical_idx = atom.bondedTo[0]
                radname = ref_template.atoms[radical_idx].name
                radname = one_atom_replace_h23_to_h12(name=radname, resname=resname)
                radical_names.append(radname)
                residue_indices.append(residues[t_idx].index)
                # find the index of the radical in the topology
                rad_idx_in_top = None
                for atom in residues[t_idx].atoms():
                    if atom.name == radname:
                        rad_idx_in_top = atom.index
                        break
                if rad_idx_in_top is None:
                    raise ValueError(f"Could not find the radical {radname} in the topology residue {residues[t_idx].index}.")

                radical_indices.append(rad_idx_in_top)
                break

    return radical_indices, radical_names, residue_indices