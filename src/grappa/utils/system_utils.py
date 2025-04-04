from typing import Optional, Union
from pathlib import Path
from openmm import System
import openmm.app
import warnings
import logging


def openmm_system_from_gmx_top(top_filepath: Union[str,Path]) -> tuple[openmm.System,openmm.app.Topology]:
    if isinstance(top_filepath,Path):
        top_filepath = top_filepath.as_posix()
    top_gmx = openmm.app.GromacsTopFile(top_filepath)

    # make sure the topology ids range from 0 to num_particles such that they correspond to the system indices:
    original_top_ids = [a.id for a in top_gmx.topology.atoms()]
    # assert they are unique:
    assert len(original_top_ids) == len(set(original_top_ids)), f"Duplicate atom ids in topology."
    # check whether they are string integers:
    assert all([i.isdigit() for i in original_top_ids]), f"Atom ids in topology are not integers."


    # convert to integers:
    original_top_ids = [int(i) for i in original_top_ids]

    # raise a warning if they are not in order:
    if original_top_ids != sorted(original_top_ids):
        warnings.warn(f"Atom ids in topology are not in order. They will be overwritten to range(0,num_atoms) to mathc indices in the openmm system. Please make sure that this is intended.")

    # if the ids are range(N,num_atoms+N) then we can just subtract N:
    if original_top_ids[0] != 0 and original_top_ids[-1] == len(original_top_ids)-1 + original_top_ids[0]:
        logging.info(f"Atom ids in the topology read form gromacs are in the range {original_top_ids[0]} to {original_top_ids[-1]}. They will be shifted to range(0,{len(original_top_ids)-1}) to match the indices in the openmm system.")

        for i, a in enumerate(top_gmx.topology.atoms()):
            a.id = i
    else:
        warnings.warn(f"Atom ids in topology are not in continuos, rising order. They will be set to range(0,num_atoms) to match the indices in the openmm system. Please make sure that this is intended.")
        for i, a in enumerate(top_gmx.topology.atoms()):
            a.id = i

    # create the system
    system_gmx = top_gmx.createSystem()
    return system_gmx, top_gmx.topology

def openmm_system_from_dict(data: dict[str,list], xml_filename: Optional[str] = None) -> tuple[openmm.System,openmm.app.Topology]:
    # check data
    required_keys = ['element','atom_name','atom_type','charge','mass','epsilon','sigma','bond','lj14scale','coulomb14scale']
    for k in required_keys:
        assert k in data.keys(),f"Key {k} missing in keys '{data.keys()}' for dictionary to create a openmm.System"
        # could construct atom_name, atom_type from elements, so no strict requirement
        # mass doesn't matter and could be a default
        # lj14scale and coulomb14scale could have defaults from amber

    # create force field
    if xml_filename is None:
        xml_filename = 'tmp_ff.xml'
    openmm_forcefield_xml_from_dict(data,xml_filename)
    ff = openmm.app.ForceField(xml_filename)

    # create topology
    openmm_topology = openmm_topology_from_dict(data)

    # create system
    openmm_system = ff.createSystem(openmm_topology)    
    return openmm_system, openmm_topology

def openmm_forcefield_xml_from_dict(data: dict, filename: str):
    lj14scale = data['lj14scale']
    coulomb14scale = data['coulomb14scale']

    # Start the XML string for ForceField
    xml_string = '<ForceField>\n'
    xml_string += '  <AtomTypes>\n'

    n_atoms = len(data['atom_name'])
    elements = data['element']
    atom_names = data['atom_name']
    atom_types = data['atom_type']
    masses = data['mass']
    charges = data['charge']
    epsilons = data['epsilon']
    sigmas = data['sigma']

    for i in range(n_atoms):
        # Construct the AtomType XML tag with just the index i for a unique name
        if atom_types[i] in atom_types[:i]:
            continue
        xml_string += f'    <Type element="{elements[i]}" name="{atom_types[i]}" class="{atom_types[i]}" mass="{masses[i]}"/>\n'

    xml_string += '  </AtomTypes>\n'
    #residues
    xml_string += '  <Residues>\n'
    xml_string += '    <Residue name="XXX">\n'
    
    #atoms
    for i in range(n_atoms):
        xml_string += f'      <Atom name="{atom_names[i]}" type="{atom_types[i]}" charge="{charges[i]}"/>\n'
    
    #bonds
    for b in data['bond']:
         xml_string += f'      <Bond atomName1="{atom_names[b[0]]}" atomName2="{atom_names[b[1]]}"/>\n'
    
    xml_string += '    </Residue>\n'
    xml_string += '  </Residues>\n'
    #nb
    xml_string += f'  <NonbondedForce coulomb14scale="{coulomb14scale}" lj14scale="{lj14scale}">\n'
    xml_string += f'    <UseAttributeFromResidue name="charge"/>\n'

    for i in range(n_atoms):
        xml_string += f'    <Atom type="{atom_types[i]}" sigma="{sigmas[i]:10.8f}" epsilon="{epsilons[i]:10.8f}"/>\n'

    xml_string += '  </NonbondedForce>\n'
    
    # Close the ForceField tag
    xml_string += '</ForceField>'
    
    # Write the XML string to a file
    with open(filename, 'w') as file:
        file.write(xml_string)

def openmm_topology_from_dict(data: dict):
    openmm_topology = openmm.app.Topology()

    # chain
    chain = openmm_topology.addChain()

    # residue
    new_residue = openmm_topology.addResidue("XXX", chain)

    # atoms
    # need atom_name, element, bond ids
    for i in range(len(data['atom_name'])):
        element = openmm.app.Element.getBySymbol(data['element'][i])
        _ = openmm_topology.addAtom(data['atom_name'][i], element, new_residue)
    atoms = list(openmm_topology.atoms())
    # bonds
    for b in data['bond']:
        _ = openmm_topology.addBond(atoms[b[0]],atoms[b[1]],type='Single')

    return openmm_topology