#%%
from pathlib import Path
import numpy as np
from ase.io import read

from kimmdy.parsing import read_top
from kimmdy.topology.topology import Topology

import openmm.app

from grappa.utils.openmm_utils import get_energies


#%%
def create_forcefield_xml(top: Topology, filename: str):
    lj14scale = top.ff.defaults[0][3]
    coulomb14scale = top.ff.defaults[0][4]

    # Start the XML string for ForceField
    xml_string = '<ForceField>\n'
    xml_string += '  <AtomTypes>\n'

    n_atoms = len(top.atoms)
    resids = []
    atom_names = []
    atom_types = []
    masses = []
    charges = []
    epsilons = []
    sigmas = []
    for a in top.atoms.values():
        a_type = top.ff.atomtypes[a.type]
        resids.append(a.resnr)
        atom_names.append(a.atom)
        atom_types.append(a.type)
        masses.append(a_type.mass)
        charges.append(a_type.charge)
        epsilons.append(a_type.epsilon)
        sigmas.append(a_type.sigma)


    for i in range(n_atoms):
        # Construct the AtomType XML tag with just the index i for a unique name
        if atom_types[i] in atom_types[:i]:
            continue
        xml_string += f'    <Type element="{atom_types[i][0]}" name="{atom_types[i]}" class="{atom_types[i]}" mass="{masses[i]}"/>\n'

    xml_string += '  </AtomTypes>\n'
    #residues
    xml_string += '  <Residues>\n'
    xml_string += '    <Residue name="XXX">\n'
    
    #atoms
    for i in range(n_atoms):
        xml_string += f'      <Atom name="{atom_names[i]}-{resids[i]}" type="{atom_types[i]}" charge="{charges[i]}"/>\n'
    
    #bonds
    for b in top.bonds.values():
        a1 = top.atoms[b.ai]
        a2 = top.atoms[b.aj]

        a1_name = f"{a1.atom}-{a1.resnr}"
        a2_name = f"{a2.atom}-{a2.resnr}"

        xml_string += f'      <Bond atomName1="{a1_name}" atomName2="{a2_name}"/>\n'
    

    xml_string += '    </Residue>\n'
    xml_string += '  </Residues>\n'
    #nb
    xml_string += f'  <NonbondedForce coulomb14scale="{coulomb14scale}" lj14scale="{lj14scale}">\n'
    xml_string += f'    <UseAttributeFromResidue name="charge"/>\n'

    for i in range(n_atoms):
        xml_string += f'    <Atom type="{atom_types[i]}" sigma="{sigmas[i]}" epsilon="{epsilons[i]}"/>\n'

   

    xml_string += '  </NonbondedForce>\n'
    
    # Close the ForceField tag
    xml_string += '</ForceField>'
    
    # Write the XML string to a file
    with open(filename, 'w') as file:
        file.write(xml_string)

def create_topology(gmx_top: Topology):
    openmm_topology = openmm.app.Topology()

    # chain
    chain = openmm_topology.addChain()

    # residue
    new_residue = openmm_topology.addResidue("XXX", chain)

    # atoms
    atoms = {}
    for a in gmx_top.atoms.values():
        atom_name = f"{a.atom}-{a.resnr}"
        element = openmm.app.Element.getBySymbol(atom_name[0])
        atoms[atom_name] = openmm_topology.addAtom(atom_name, element, new_residue)

    # bonds
    for b in gmx_top.bonds.values():
        a1 = top.atoms[b.ai]
        a2 = top.atoms[b.aj]

        a1_name = f"{a1.atom}-{a1.resnr}"
        a2_name = f"{a2.atom}-{a2.resnr}"
        _ = openmm_topology.addBond(atoms[a1_name],atoms[a2_name],type='Single')

    return openmm_topology

#%%
gaussian_dir = Path('/hits/fast/mbm/hartmaec/datasets_v2/AA_opt_rad')
gaussian_dir = Path('/hits/fast/mbm/hartmaec/workdir/grappa_simulations/radical_datasets/test_dataset/QM')
top_dir = Path('/hits/fast/mbm/hartmaec/workdir/grappa_simulations/radical_datasets/test_dataset/top')

#%%
structures = []
gaussian_files = list(gaussian_dir.glob('*CA_opt.log')) + list(gaussian_dir.glob('*.out'))
for file in gaussian_files:
    structures.append(read(file,index=':'))

QM_at_num = structures[0][0].arrays['numbers']
# %%
for top_file in top_dir.glob('*top'):
    top = Topology(read_top(top_file))
    # fudgeLJ and fudgeQQ assignment is not robust like this
    atomic_numbers = []
    for a in top.atoms.values():
        a_type = top.ff.atomtypes[a.type]
        atomic_numbers.append(a_type.at_num)

print(f"Atom numbers match: {all(QM_at_num == [int(i) for i in atomic_numbers])}")

# %%
xml_filename = 'tmp_ff.xml'
create_forcefield_xml(top,xml_filename)
# %%

ff = openmm.app.ForceField(xml_filename)
# %%
openmm_topology = create_topology(top)
# %%
system = ff.createSystem(openmm_topology,nonbondedMethod=openmm.app.NoCutoff)
# %%
xyz = np.array([x.arrays['positions'] for x in structures[0]])
energy,force = get_energies(system,xyz)
# %%
