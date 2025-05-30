OPENMM_WATER_RESIDUES = ["HOH", "WAT", "TIP3", "TIP4", "TIP5", "TIP3P", "TIP4P", "TIP5P", "SPC", "SPC/E", "SPCE", "SPC-FW", "SPC-HW", "SPC-HFW", "SPC-HF"]
OPENMM_ION_RESIDUES = ["CL", "NA", "K", "MG", "CA", "ZN", "FE", "CU", "F", "BR", "I", "CL-", "NA+", "K+", "MG2+", "CA2+", "ZN2+", "FE2+", "FE3+", "CU2+", "CU1+", "F-", "BR-", "I-"]

# The following conditional imnports are only for pylint, we import the packages in each function call again. otherwise, it will not work if one first installs grappa and only then openmm.
import importlib.util
if importlib.util.find_spec('openmm') is not None:
    import openmm
    from openmm.app import PDBFile
    from openmm.app import ForceField, Topology
    from openmm.unit import dalton
    from openmm.openmm import PeriodicTorsionForce

from io import StringIO
from collections import defaultdict
from grappa.utils import get_repo_dir
import numpy as np
from typing import Union, Dict, List
from pathlib import Path
import tempfile
from grappa.constants import get_grappa_units_in_openmm, get_openmm_units
from typing import Tuple
import grappa.data
import copy
import warnings
import logging

# Define a custom filter to raise the warning only once
warnings.simplefilter("once")



def get_subtopology(topology:'openmm.app.topology.Topology', exclude_residues:List[str]=None)->'openmm.app.topology.Topology':
    """
    Returns a sub-topology of the given topology, excluding certain residues with names given in exclude_residues.
    The atom.id of the atoms in the sub-topology is the same as the atom.index in the original topology.
    """
    import openmm.app.topology

    assert isinstance(topology, openmm.app.topology.Topology), f"Expected openmm.app.topology.Topology, but got {type(topology)}"

    if exclude_residues is None:
        return topology
    
    # create a new topology:
    new_topology = openmm.app.topology.Topology()

    new_topol_idx = {} # maps the old atom index to the new atom index

    ignored_residues = defaultdict(int)

    # add a dummy chain and residue:
    new_chain = new_topology.addChain()
    new_residue = new_topology.addResidue('DUM', new_chain)

    # add all atoms ensuring that their atom.id is the index in the original topology
    for atom in topology.atoms():
        if atom.residue.name not in exclude_residues:
            new_topology.addAtom(atom.name, atom.element, new_residue, id=atom.index)
            new_topol_idx[atom.index] = new_topology.getNumAtoms() - 1
        else:
            ignored_residues[atom.residue.name] += 1

    if ignored_residues:
        logging.info(f"Ignored atoms of the following residues:\n{dict(ignored_residues)}")

    new_atoms = list(new_topology.atoms())

    # add all bonds:
    # we only add bonds where both atoms are in the new topology
    # obtain the old indices, map to new indices, and pick the atoms from the new topology
    for bond in topology.bonds():
        if all([atom.index in new_topol_idx.keys() for atom in bond]):
            new_topology.addBond(new_atoms[new_topol_idx[bond[0].index]], new_atoms[new_topol_idx[bond[1].index]])

    return new_topology


def get_energies(openmm_system: 'openmm.System', xyz:np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    """
    Returns energies, forces. in units kcal/mol and kcal/mol/angstroem.
    Assume that xyz is in angstroem and has shape (num_confs, num_atoms, 3).
    """
    import openmm
    from openmm import unit

    assert len(xyz.shape) == 3, f"xyz must have shape (num_confs, num_atoms, 3), but got {xyz.shape}"
    assert xyz.shape[1] == openmm_system.getNumParticles(), f"Number of atoms in xyz ({xyz.shape[1]}) does not match number of atoms in system ({openmm_system.getNumParticles()})"
    assert xyz.shape[2] == 3, f"xyz must have shape (num_confs, num_atoms, 3), but got {xyz.shape}"

    if xyz.shape[0] == 0:
        return np.array([]).astype(np.float32), np.zeros(xyz.shape).astype(np.float32)

    # create a context:
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(openmm_system, integrator)

    energies = []
    forces = []
    # set positions:
    for pos in xyz:
        context.setPositions(unit.Quantity(pos, unit.angstrom))
        state = context.getState(getEnergy=True, getForces=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        forces_ = state.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole/unit.angstrom)
        energies.append(energy)
        forces.append(forces_)

    return np.array(energies), np.array(forces)


def remove_forces_from_system(system:'openmm.System', remove:Union[List[str], str]=None, keep:List[str]=None, info=False)->'openmm.System':
    """
    Modifies the OpenMM system by removing forces according to the 'remove' and 'keep' lists.
    Forces are identified by their class name. E.g. to remove all nonbonded forces, use remove='nonbonded', to only keep nonbonded forces, use keep='nonbonded'.

    Parameters:
    - system: The OpenMM System object to modify.
    - remove: A list of strings. Forces with class names containing any of these strings will be removed.
    - keep: A list of strings. If not None, only forces with class names containing these strings will be kept.

    Returns:
    - system: The modified OpenMM System object.
    """

    if not isinstance(remove, list):
        remove = [remove]

    # First, identify the indices of the forces to remove
    forces_to_remove = []
    for i, force in enumerate(system.getForces()):
        force_name = force.__class__.__name__.lower()
        assert force_name in ['nonbondedforce', 'harmonicbondforce', 'harmonicangleforce', 'periodictorsionforce', 'cmaptorsionforce', 'custombondforce', 'customangleforce', 'customtorsionforce', 'customnonbondedforce', 'cmmotionremover','customgbforce'], f"Force found in openmm system ({force_name}) that is not implemented in remove_forces_from_system"
        if force_name == 'customgbforce':
            print(f"WARNING: Non-gaff force in force field: {force_name}")
        if keep is not None:
            assert isinstance(keep, list), f"Expected keep to be a list, but got {type(keep)}"
            assert len(keep) > 0, "Expected keep to be a list of strings, but got an empty list."
            if not any([k.lower() in force_name for k in keep]):
                forces_to_remove.append(i)
                if info:
                    print(f"Removing force {force_name}")
        elif remove is not None:
            if any([k.lower() in force_name for k in remove]):
                forces_to_remove.append(i)
                if info:
                    print(f"Removing force {force_name}")

    # Remove the forces by index, in reverse order to not mess up the indices
    for i in reversed(forces_to_remove):
        system.removeForce(i)

    return system


def set_partial_charges(system:'openmm.System', partial_charges:Union[list, np.ndarray])->'openmm.System':
    """
    Set partial charges of a system. The charge must be in units of elementary charge.
    """
    import openmm
    # get the nonbonded force (behaves like a reference not a copy!):
    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            if not nonbonded_force is None:
                raise ValueError("More than one nonbonded force found.")
            nonbonded_force = force
            
    if nonbonded_force is None:
        raise ValueError("No nonbonded force found.")
    
    # set the charges:
    if len(partial_charges) != nonbonded_force.getNumParticles():
        raise ValueError("Number of partial charges does not match number of particles.")
    
    for i, charge in enumerate(partial_charges):
        # get the parameters:
        _, sigma, epsilon = nonbonded_force.getParticleParameters(i)
        # set the charge:
        nonbonded_force.setParticleParameters(i, charge=charge, sigma=sigma, epsilon=epsilon)

    return system

def get_partial_charges(system:'openmm.System')->np.ndarray:
    """
    Returns the partial charges of the system in units of elementary charge.
    """
    import openmm
    # get the nonbonded force (behaves like a reference not a copy!):
    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            if not nonbonded_force is None:
                raise ValueError("More than one nonbonded force found.")
            nonbonded_force = force
            
    if nonbonded_force is None:
        raise ValueError("No nonbonded force found.")
    
    charges = []
    for i in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
        charges.append(charge.value_in_unit(openmm.unit.elementary_charge))

    charges = np.array(charges)
    return charges


def write_to_system(system:'openmm.System', parameters:grappa.data.Parameters)->'openmm.System':
    """
    Writes bonded parameters in an openmm system. For interactions that are already present in the system, overwrite the parameters; otherwise add the interaction to the system. The forces, however, must be already present in the system.
    The ids of the atoms, bonds, etc in the parameters object must be the same as the system indices.
    The ids must n0t necessarily run from 0 to N-1, they can also represent a subset of the system indices.
    """

    # handle units:
    from openmm.unit import Quantity
    import openmm

    grappa_units = get_grappa_units_in_openmm()
    BOND_K_UNIT = grappa_units['BOND_K']
    BOND_EQ_UNIT = grappa_units['BOND_EQ']
    ANGLE_K_UNIT = grappa_units['ANGLE_K']
    ANGLE_EQ_UNIT = grappa_units['ANGLE_EQ']
    TORSION_K_UNIT = grappa_units['TORSION_K']
    TORSION_PHASE_UNIT = grappa_units['TORSION_PHASE']


    bonds = parameters.bonds
    angles = parameters.angles
    impropers = parameters.impropers
    propers = parameters.propers


    bond_ks = Quantity(parameters.bond_k, unit=BOND_K_UNIT)
    bond_eqs = Quantity(parameters.bond_eq, unit=BOND_EQ_UNIT)
    angle_ks = Quantity(parameters.angle_k, unit=ANGLE_K_UNIT)
    angle_eqs = Quantity(parameters.angle_eq, unit=ANGLE_EQ_UNIT)
    improper_ks = Quantity(parameters.improper_ks, unit=TORSION_K_UNIT)
    improper_phases = Quantity(parameters.improper_phases, unit=TORSION_PHASE_UNIT)
    proper_ks = Quantity(parameters.proper_ks, unit=TORSION_K_UNIT)
    proper_phases = Quantity(parameters.proper_phases, unit=TORSION_PHASE_UNIT)

    assert np.all(parameters.proper_ks >= 0)
    assert np.all(parameters.improper_ks >= 0)

    # create a dictionary because we will need lookups and dict lookup is more efficient than list.index (these are shallow copies)
    bond_lookup = {tuple(b):(bond_ks[i], bond_eqs[i]) for i, b in enumerate(bonds)}
    angle_lookup = {tuple(a): (angle_ks[i], angle_eqs[i]) for i, a in enumerate(angles)}

    ordered_torsions = {tuple(sorted(p)) for p in list([imp for imp in impropers]) + list([prop for prop in propers])}

    # loop through the system forces, for all parameters in the parameters object, overwrite the system parameters if present, otherwise add the interaction.
    # Note that if the system contains bonds/angles/... between atoms that are not in the parameters object, these bonds/angles... will be untouched, i.e. kept as they are.
    # in each step, first transform the parameter id to the system index.

    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                # Get the atom indices and existing parameters
                atom1, atom2, length, k = force.getBondParameters(i)

                # try both orderings:
                bond_param = bond_lookup.get((atom1, atom2), None)
                if bond_param is None:
                    bond_param = bond_lookup.get((atom2, atom1), None)
                    if not bond_param is None:
                        bond_lookup.pop((atom2, atom1))
                else:
                    bond_lookup.pop((atom1, atom2))

                if not bond_param is None:
                    # Update the parameters
                    new_k, new_length = bond_param
                    force.setBondParameters(i, atom1, atom2, new_length, new_k)


        elif isinstance(force, openmm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                atom1, atom2, atom3, _, _ = force.getAngleParameters(i)

                angle_param = angle_lookup.pop((atom1, atom2, atom3), None)
                if angle_param is None:
                    angle_param = angle_lookup.pop((atom3, atom2, atom1), None)

                if not angle_param is None:
                    new_k, new_angle = angle_param
                    force.setAngleParameters(i, atom1, atom2, atom3, new_angle, new_k)


        # check whether torsion is contained in both proper or improper. if so, set its k to zero, effectively removing the force.
        if isinstance(force, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                atom1, atom2, atom3, atom4, periodicity, phase, k = force.getTorsionParameters(i)

                # Check in proper and improper torsions
                if tuple(sorted((atom1, atom2, atom3, atom4))) in ordered_torsions:
                    # Set k to zero to effectively remove from the system. We will add another torsion force later.
                    force.setTorsionParameters(i, atom1, atom2, atom3, atom4, periodicity, phase, 0)

    
        # now add the bonds and angles that have not been added yet as new forces.
        # also add a new torsion force, one for proper, one for improper.

    # Adding remaining bonds
    if bond_lookup:
        new_bond_force = openmm.HarmonicBondForce()
        for bond, params in bond_lookup.items():
            new_bond_force.addBond(bond[0], bond[1], length=params[1], k=params[0])
        system.addForce(new_bond_force)

    # Adding remaining angles
    if angle_lookup:
        new_angle_force = openmm.HarmonicAngleForce()
        for angle, params in angle_lookup.items():
            new_angle_force.addAngle(angle[0], angle[1], angle[2], angle=params[1], k=params[0])
        system.addForce(new_angle_force)

    # Adding all torsions:
    proper_torsion_force = openmm.PeriodicTorsionForce()
    for i, torsion in enumerate(propers):
        for n in range(len(proper_ks[i])):
            if proper_ks[i][n].value_in_unit(TORSION_K_UNIT) != 0.:
                proper_torsion_force.addTorsion(torsion[0], torsion[1], torsion[2], torsion[3], periodicity=n+1, phase=proper_phases[i][n], k=proper_ks[i][n])

    # Adding all impropers:
    improper_torsion_force = openmm.PeriodicTorsionForce()
    for i, torsion in enumerate(impropers):
        for n in range(len(improper_ks[i])):
            if improper_ks[i][n].value_in_unit(TORSION_K_UNIT) != 0.:
                improper_torsion_force.addTorsion(torsion[0], torsion[1], torsion[2], torsion[3], periodicity=n+1, phase=improper_phases[i][n], k=improper_ks[i][n])

    system.addForce(proper_torsion_force)
    system.addForce(improper_torsion_force)

    return system



def topology_from_pdb(pdbstring:str)->'openmm.app.topology.Topology':
    """
    Returns an openmm topology from a pdb string in which the lines are separated by '\n'.
    """
    return get_pdb(pdbstring).getTopology()


def get_openmm_forcefield(name:str, *args, **kwargs)->'ForceField':
    """
    The name can be given either with or without .xml ending. Possible names are all openmm forcefield names and:
    - charmm36 (capable of parametrizing ACE and NME caps)
    - amber99sbildn* or amber99sbildn-star (amber99sbildn with HYP and DOP residue type)
    - any standard openmm forcefield
    """
    from openmm.app import ForceField

    if name.endswith('.xml'):
        name = name[:-4]
    
    if name in ['charmm36', 'charmm36-jul2022']:
        warnings.warn("The charmm36 forcefield implemented in grappa for dealing with ACE and NME has a faulty improper contribution. Only use the other contributions.", UserWarning)
        ff_path = get_repo_dir() / "src" / "grappa" / "utils" / "classical_forcefields" / "charmm36-jul2022.xml"
        return ForceField(str(ff_path), *args, **kwargs)

    elif name == 'amber99sbildn*' or name == 'amber99sbildn-star':
        from grappa.utils import hyp_dop_utility

        ff_path = get_repo_dir() / "src" / "grappa" / "utils" / "classical_forcefields" / "amber99sbildn-star_.xml"

        class HypDopOpenmmForceField:
            """
            Modify the createSystem method because openmm.PDBFile cannot read HYP and DOP residues properly.
            """
            def __init__(self, ff_path:Union[Path, str], *args, **kwargs):
                self.ff = ForceField(str(ff_path), *args, **kwargs)
            
            def createSystem(self, topology, *args, **kwargs):
                """
                Create the system. This method is overwritten because openmm.PDBFile cannot read HYP and DOP residues properly.
                """
                # add bonds that were not written to the topology yet:

                topology = hyp_dop_utility.add_bonds(topology)
                return self.ff.createSystem(topology, *args, **kwargs)
                
        return HypDopOpenmmForceField(str(ff_path), *args, **kwargs)

    else:
        return ForceField(name+'.xml', *args, **kwargs)


def get_nonbonded_contribution(openmm_system:'openmm.System', xyz):
    return get_contribution(openmm_system, xyz, keywords=['nonbonded','customgbforce'])

def get_bond_contribution(openmm_system:'openmm.System', xyz:np.ndarray):
    return get_contribution(openmm_system, xyz, keywords=['bondforce'])

def get_angle_contribution(openmm_system:'openmm.System', xyz:np.ndarray):
    return get_contribution(openmm_system, xyz, keywords=['angle'])

def get_torsion_contribution(openmm_system:'openmm.System', xyz:np.ndarray):
    """
    This is Proper + Improper contribution!
    """
    return get_contribution(openmm_system, xyz, keywords=['torsion'])


def get_contribution(openmm_system:'openmm.System', xyz:np.ndarray, force:Union[str,List[str]]=None, keywords:List[str]=[])->Tuple[np.ndarray, np.ndarray]:
    """
    Create a deep copy of the openmm system and remove all forces except the force given in the argument. Then calculate the energy and gradient of the states in that system.
    keywords: if any low(keywords) in force, keep the force.
    """
    openmm_system = copy.deepcopy(openmm_system)

    keep = force if force is not None else []
    keep = [keep] if isinstance(keep, str) else keep
    assert isinstance(keep, list), f"Expected force to be a list, but got {type(force)}"
    
    if keywords is not None:
        assert isinstance(keywords, list), f"Expected keywords to be a list, but got {type(keywords)}"
        keep += keywords

    openmm_system = remove_forces_from_system(openmm_system, keep=keep)

    energy, gradient = get_energies(openmm_system=openmm_system, xyz=xyz)
    gradient = -gradient # the reference gradient is the negative of the force

    return energy, gradient



def get_improper_contribution(openmm_system:'openmm.System', xyz:np.ndarray, molecule):
        """
        Only works if the impropers are given as PeriodicTorsionForce in the openmm system.
        """
        import openmm

        openmm_system = copy.deepcopy(openmm_system)

        # calculate the contribution from improper torsions in the system:
        # remove all forces but periodic torsions (we assume that impropers are periodic torsions)
        openmm_system = remove_forces_from_system(openmm_system, keep=['PeriodicTorsionForce'])

        # get a list of sets of improper torsion tuples:
        improper_set = {tuple(sorted(t)) for t in molecule.impropers}

        # set all ks to zero that are not impropers:
        for force in openmm_system.getForces():
            if not isinstance(force, openmm.PeriodicTorsionForce):
                raise NotImplementedError(f"Removed all but PeriodicTorsionForce, but found a different force: {force.__class__.__name__}")
            for i in range(force.getNumTorsions()):
                atom1, atom2, atom3, atom4, periodicity, phase, k = force.getTorsionParameters(i)
                if not tuple(sorted((molecule.atoms[atom1], molecule.atoms[atom2], molecule.atoms[atom3], molecule.atoms[atom4]))) in improper_set:
                    force.setTorsionParameters(i, atom1, atom2, atom3, atom4, periodicity, phase, 0)


        # get energy and gradient. these are now only sourced from improper torsions.
        improper_energy, improper_gradient = get_energies(openmm_system=openmm_system, xyz=xyz)
        improper_gradient = -improper_gradient # the reference gradient is the negative of the force

        return improper_energy, improper_gradient


def get_pdb(pdb_string:str)->'PDBFile':
    """
    Returns an openmm PDBFile from a pdb string in which the lines are separated by '\n'.
    """
    from openmm.app import PDBFile
    with tempfile.TemporaryDirectory() as tmp:
        pdbpath = str(Path(tmp)/'pep.pdb')
        with open(pdbpath, "w") as pdb_file:
            pdb_file.write(pdb_string)
        openmm_pdb = PDBFile(pdbpath)

    return openmm_pdb

def get_forcefield_xml_base(atom_names, atom_types, charges, masses, elements, bonds):

    xml_string = '<ForceField>\n'
    xml_string += '  <AtomTypes>\n'

    n_atoms = len(atom_names)

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
    for b in bonds:
         xml_string += f'      <Bond atomName1="{b[0]}" atomName2="{b[1]}"/>\n'
    
    xml_string += '    </Residue>\n'
    xml_string += '  </Residues>\n'

    # end force field definition
    xml_string += '</ForceField>'

    return xml_string


def get_nonbonded_force_xml(atom_types, epsilons, sigmas, lj14scale, coulomb14scale):
    xml_string = ''
    n_atoms = len(atom_types)

    xml_string += f'  <NonbondedForce coulomb14scale="{coulomb14scale}" lj14scale="{lj14scale}">\n'
    xml_string += f'    <UseAttributeFromResidue name="charge"/>\n'

    for i in range(n_atoms):
        xml_string += f'    <Atom type="{atom_types[i]}" sigma="{sigmas[i]:10.8f}" epsilon="{epsilons[i]:10.8f}"/>\n'

    xml_string += '  </NonbondedForce>\n'

    return xml_string


def get_bond_force_xml(bond_idxs, bond_eqs, bond_ks):
    xml = '  <HarmonicBondForce>\n'
    for (i, j), eq, k in zip(bond_idxs, bond_eqs, bond_ks):
        xml += f'    <Bond class1="{i}" class2="{j}" length="{eq:.8f}" k="{k:.8f}"/>\n'
    xml += '  </HarmonicBondForce>\n'
    return xml


def get_angle_force_xml(angle_idxs, angle_eqs, angle_ks):
    xml = '  <HarmonicAngleForce>\n'
    for (i, j, k_), eq, k_angle in zip(angle_idxs, angle_eqs, angle_ks):
        xml += f'    <Angle class1="{i}" class2="{j}" class3="{k_}" angle="{eq:.8f}" k="{k_angle:.8f}"/>\n'
    xml += '  </HarmonicAngleForce>\n'
    return xml


def get_torsion_force_xml(torsion_idxs, torsion_phases, torsion_ks, name='Proper'):
    xml = '  <PeriodicTorsionForce>\n'
    for (i, j, k, l), ks, phases in zip(torsion_idxs, torsion_ks, torsion_phases):
        if all([k_val == 0 for k_val in ks]):
            continue
        # if more than one term:
        if sum([k_val != 0 for k_val in ks]) > 1:
            xml += f'    <{name} class1="{i}" class2="{j}" class3="{k}" class4="{l}"\n'
            term_counter = 1
            for n, (k_val, phase) in enumerate(zip(ks, phases)):
                if k_val == 0:
                    continue
                xml += f'        periodicity{term_counter}="{n+1}" phase{term_counter}="{phase:.8f}" k{term_counter}="{k_val:.8f}"'
                if n < len(ks) - 1:
                    xml += '\n'
                term_counter += 1
            xml += '/>\n'
        else:
            # if only one term:
            # find non-zero term:
            for n, (k_val, phase) in enumerate(zip(ks, phases)):
                if k_val != 0:
                    xml += f'    <{name} class1="{i}" class2="{j}" class3="{k}" class4="{l}" periodicity1="{n+1}" phase1="{phase:.8f}" k1="{k_val:.8f}"/>\n'
                    break
    xml += '  </PeriodicTorsionForce>\n'
        
    return xml


def xml_from_lists(atom_names, atom_types, charges, masses, epsilons, sigmas,
    elements, lj14scale, coulomb14scale, bonds,
    bond_idxs=None, bond_eqs=None, bond_ks=None,
    angle_idxs=None, angle_eqs=None, angle_ks=None,
    proper_idxs=None, proper_phases=None, proper_ks=None,
    improper_idxs=None, improper_phases=None, improper_ks=None):
    xml = get_forcefield_xml_base(
        atom_names=atom_names,
        atom_types=atom_types,
        charges=charges,
        masses=masses,
        elements=elements,
        bonds=bonds
    )
    assert xml.endswith('</ForceField>'), "Internal error. XML string should end with '</ForceField>'"
    xml = xml[:-len('</ForceField>')]

    xml += get_nonbonded_force_xml(
        atom_types=atom_types,
        epsilons=epsilons,
        sigmas=sigmas,
        lj14scale=lj14scale,
        coulomb14scale=coulomb14scale
    )

    if bond_idxs:
        xml += get_bond_force_xml(
            bond_idxs=bond_idxs,
            bond_eqs=bond_eqs,
            bond_ks=bond_ks
        )

    if angle_idxs:
        xml += get_angle_force_xml(
            angle_idxs=angle_idxs,
            angle_eqs=angle_eqs,
            angle_ks=angle_ks
        )

    if proper_idxs:
        xml += get_torsion_force_xml(
            torsion_idxs=proper_idxs,
            torsion_phases=proper_phases,
            torsion_ks=proper_ks,
            name='Proper'
        )

    if improper_idxs:
        xml += get_torsion_force_xml(
            torsion_idxs=improper_idxs,
            torsion_phases=improper_phases,
            torsion_ks=improper_ks,
            name='Improper'
        )

    xml += '</ForceField>'
    return xml


def create_forcefield(topology, parameters, coulomb_fudge=None, lj_fudge=None):
    grappa_units = get_grappa_units_in_openmm()
    openmm_units = get_openmm_units()

    if coulomb_fudge is None:
        coulomb_fudge = parameters.coulomb_fudge if parameters.coulomb_fudge is not None else 0.833
    if lj_fudge is None:
        lj_fudge = parameters.lj_fudge if parameters.lj_fudge is not None else 0.5

    atom_names, atom_types, charges, masses, elements = [], [], [], [], []

    for i, atom in enumerate(topology.atoms()):
        atom_names.append(str(i))
        atom_types.append(str(i))
        q = parameters.partial_charges[i] * grappa_units["CHARGE"]
        charges.append(q.value_in_unit(openmm_units["CHARGE"]))
        masses.append(atom.element.mass.value_in_unit(dalton))
        elements.append(atom.element.symbol)

    epsilons = [
        (parameters.epsilons[i] * grappa_units["EPSILON"]).value_in_unit(openmm_units["EPSILON"])
        for i in range(len(parameters.atoms))
    ]
    sigmas = [
        (parameters.sigmas[i] * grappa_units["SIGMA"]).value_in_unit(openmm_units["SIGMA"])
        for i in range(len(parameters.atoms))
    ]

    bonds = [(a1.index, a2.index) for a1, a2 in topology.bonds()]

    bond_idxs, bond_eqs, bond_ks = [], [], []
    if hasattr(parameters, "bonds") and parameters.bonds is not None:
        for (i, j), k, eq in zip(parameters.bonds, parameters.bond_k, parameters.bond_eq):
            bond_idxs.append((i, j))
            bond_eqs.append((eq * grappa_units["BOND_EQ"]).value_in_unit(openmm_units["BOND_EQ"]))
            bond_ks.append((k * grappa_units["BOND_K"]).value_in_unit(openmm_units["BOND_K"]))

    angle_idxs, angle_eqs, angle_ks = [], [], []
    if hasattr(parameters, "angles") and parameters.angles is not None:
        for (i, j, k_), k_angle, eq in zip(parameters.angles, parameters.angle_k, parameters.angle_eq):
            angle_idxs.append((i, j, k_))
            angle_eqs.append((eq * grappa_units["ANGLE_EQ"]).value_in_unit(openmm_units["ANGLE_EQ"]))
            angle_ks.append((k_angle * grappa_units["ANGLE_K"]).value_in_unit(openmm_units["ANGLE_K"]))

    proper_idxs, proper_phases, proper_ks = [], [], []
    if hasattr(parameters, "propers") and parameters.propers is not None:
        for (i, j, k, l), ks, phases in zip(parameters.propers, parameters.proper_ks, parameters.proper_phases):
            proper_idxs.append((i, j, k, l))
            proper_phases.append([(p * grappa_units["TORSION_PHASE"]).value_in_unit(openmm_units["TORSION_PHASE"]) for p in phases])
            proper_ks.append([(k_ * grappa_units["TORSION_K"]).value_in_unit(openmm_units["TORSION_K"]) for k_ in ks])

    improper_idxs, improper_phases, improper_ks = [], [], []
    if hasattr(parameters, "impropers") and parameters.impropers is not None:
        for (i, j, k, l), ks, phases in zip(parameters.impropers, parameters.improper_ks, parameters.improper_phases):
            improper_idxs.append((i, j, k, l))
            improper_phases.append([(p * grappa_units["TORSION_PHASE"]).value_in_unit(openmm_units["TORSION_PHASE"]) for p in phases])
            improper_ks.append([(k_ * grappa_units["TORSION_K"]).value_in_unit(openmm_units["TORSION_K"]) for k_ in ks])

    # xml doesnt work for properly for improper torsions, thus handle them separately

    xml_string = xml_from_lists(
        atom_names=atom_names,
        atom_types=atom_types,
        charges=charges,
        masses=masses,
        epsilons=epsilons,
        sigmas=sigmas,
        elements=elements,
        lj14scale=lj_fudge,
        coulomb14scale=coulomb_fudge,
        bonds=bonds,
        bond_idxs=bond_idxs,
        bond_eqs=bond_eqs,
        bond_ks=bond_ks,
        angle_idxs=angle_idxs,
        angle_eqs=angle_eqs,
        angle_ks=angle_ks,
        proper_idxs=None,
        proper_phases=None,
        proper_ks=None,
        improper_idxs=None,
        improper_phases=None,
        improper_ks=None
    )

    return xml_string, proper_idxs, proper_phases, proper_ks, improper_idxs, improper_phases, improper_ks


def get_single_res_top(topology:'openmm.app.topology.Topology')->'openmm.app.topology.Topology':
    """
    Returns a new topology with a single residue containing all atoms from the original topology.
    The atom.id of the atoms in the new topology is the same as the atom.index in the original topology.
    This is useful for creating a topology that can be used with the GrappaForceField, which requires a single residue.
    """
    topo = Topology()
    chain = topo.addChain()
    res = topo.addResidue("XXX", chain)

    atom_map = {}

    original_atoms = list(topology.atoms())
    for i, atom in enumerate(original_atoms):
        new_atom = topo.addAtom(str(i), atom.element, res)
        atom_map[atom] = new_atom

    for a1, a2 in topology.bonds():
        if a1 in atom_map and a2 in atom_map:
            topo.addBond(atom_map[a1], atom_map[a2])

    return topo


if importlib.util.find_spec('openmm') is None:
    class GrappaForceField:
        """
        Dummy ForceField class that has an own atom type for each atom. allows for proper and improper torsions to be defined separately.
        """
        def __init__(self, xml_file, *args, **kwargs):
            raise ImportError("OpenMM is not installed. Please install OpenMM to use the GrappaForceField class.")
        
else:
    class GrappaForceField(ForceField):
        def __init__(self, xml_file, proper_idxs=None, proper_phases=None, proper_ks=None, improper_idxs=None, improper_phases=None, improper_ks=None):
            """
            Dummy ForceField class that has an own atom type for each atom. allows for proper and improper torsions to be defined separately.
            """
            super().__init__(xml_file)
            self.proper_idxs = proper_idxs
            self.proper_phases = proper_phases
            self.proper_ks = proper_ks
            self.improper_idxs = improper_idxs
            self.improper_phases = improper_phases
            self.improper_ks = improper_ks

        @classmethod
        def from_parameters(cls, parameters, topology, coulomb_fudge=None, lj_fudge=None):
            """
            Create a GrappaForceField from parameters and topology.
            Parameters:
            - parameters: grappa.data.Parameters object containing the force field parameters.
            - topology: openmm.app.topology.Topology object containing the topology of the system.
            - coulomb_fudge: Coulomb fudge factor
            - lj_fudge: Lennard-Jones fudge factor
            Returns:
            - An instance of GrappaForceField.
            """

            xml_string, proper_idxs, proper_phases, proper_ks, improper_idxs, improper_phases, improper_ks = create_forcefield(
                topology=topology,
                parameters=parameters,
                coulomb_fudge=coulomb_fudge,
                lj_fudge=lj_fudge
            )

            return cls(StringIO(xml_string),
                       proper_idxs=proper_idxs,
                       proper_phases=proper_phases,
                       proper_ks=proper_ks,
                       improper_idxs=improper_idxs,
                       improper_phases=improper_phases,
                       improper_ks=improper_ks)

        def createSystem(
            self,
            topology,
            flexibleConstraints=False,
            drudeMass=None,
            *args,
            **kwargs
        ):
            # the arguments flexibleConstraints and drudeMass need to be called explicitly for some reason
            topo = get_single_res_top(topology)
            system = super().createSystem(
                topology=topo,
                flexibleConstraints=flexibleConstraints,
                drudeMass=drudeMass,
                *args,
                **kwargs
            )

            system = self.add_torsions(system)
            return system
        
        def add_torsions(self, system):
            """
            Add proper and improper torsions to the system.
            """
            if self.proper_idxs is not None:
                torsion_force = PeriodicTorsionForce()
                for (i, j, k, l), phases, ks in zip(self.proper_idxs, self.proper_phases, self.proper_ks):
                    for n, (phase, k_) in enumerate(zip(phases, ks)):
                        if k_ == 0:
                            continue
                        torsion_force.addTorsion(i, j, k, l, n + 1, phase, k_)
            
            if self.improper_idxs is not None:
                improper_force = PeriodicTorsionForce()
                for (i, j, k, l), phases, ks in zip(self.improper_idxs, self.improper_phases, self.improper_ks):
                    for n, (phase, k_) in enumerate(zip(phases, ks)):
                        if k_ == 0:
                            continue
                        improper_force.addTorsion(i, j, k, l, n + 1, phase, k_)

            if self.proper_idxs is not None:
                system.addForce(torsion_force)
            if self.improper_idxs is not None:
                system.addForce(improper_force)

            return system

