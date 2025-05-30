#%%
from grappa.utils.data_utils import get_moldata_path
from grappa.utils.openmm_utils import get_pdb
from openmm.app import ForceField
from grappa.data import MolData, Molecule, Parameters

# %%
paths = list(get_moldata_path("spice-dipeptide-amber99").glob("*.npz"))
moldata = MolData.load(paths[0])
pdbfile = get_pdb(moldata.pdb)
forcefield = ForceField("amber99sb.xml")
system = forcefield.createSystem(pdbfile.topology)
top = pdbfile.topology

#%%
molecule = Molecule.from_openmm_system(openmm_system=system, openmm_topology=pdbfile.topology)
params = Parameters.from_openmm_system(openmm_system=system, mol=molecule)

assert list(range(len(params.atoms))) == params.atoms
# %%
top = pdbfile.topology
for i, atom in enumerate(top.atoms()):
    mass = atom.element.mass
    symbol = atom.element.symbol
    # print(mass)
#%%

def xml_from_lists(atom_names, atom_types, charges, masses, epsilons, sigmas, elements, lj14scale, coulomb14scale, bonds):

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
    #nb
    xml_string += f'  <NonbondedForce coulomb14scale="{coulomb14scale}" lj14scale="{lj14scale}">\n'
    xml_string += f'    <UseAttributeFromResidue name="charge"/>\n'

    for i in range(n_atoms):
        xml_string += f'    <Atom type="{atom_types[i]}" sigma="{sigmas[i]:10.8f}" epsilon="{epsilons[i]:10.8f}"/>\n'

    xml_string += '  </NonbondedForce>\n'
    
    # Close the ForceField tag
    xml_string += '</ForceField>'
    return xml_string

from openmm.app import ForceField, Topology
from openmm.app import element as elem
from xml.etree.ElementTree import Element as XMLElement, SubElement, tostring
from openmm.unit import dalton
from grappa.constants import get_grappa_units_in_openmm, get_openmm_units
import copy
from io import StringIO

from simtk.openmm.app import ForceField
from io import StringIO
import copy

class GrappaForceField(ForceField):
    def __init__(self, xml_string):
        super().__init__(StringIO(xml_string))

    def createSystem(self, topology, *args, **kwargs):
        topo = copy.deepcopy(topology)
        for i, atom in enumerate(topo.atoms()):
            atom.name = str(i)
            atom.type = str(i)
            atom.residue.name = "XXX"  # match the <Residue name="XXX"> in XML
        # now also 
        return topo
        # return super().createSystem(topo, *args, **kwargs)

def create_forcefield(topology, parameters, coulomb_fudge=None, lj_fudge=None):
    grappa_units = get_grappa_units_in_openmm()
    openmm_units = get_openmm_units()

    if coulomb_fudge is None:
        coulomb_fudge = parameters.coulomb_fudge if parameters.coulomb_fudge is not None else 0.833
    if lj_fudge is None:
        lj_fudge = parameters.lj_fudge if parameters.lj_fudge is not None else 0.5

    atom_names = []
    atom_types = []
    charges = []
    masses = []
    elements = []

    for i, atom in enumerate(topology.atoms()):
        assert atom.index == i, f"Atom index mismatch: {atom.index} != {i}"
        atom_names.append(str(i))
        atom_types.append(str(i))
        q_grappa = parameters.partial_charges[i] * grappa_units["CHARGE"]
        charges.append(q_grappa.value_in_unit(openmm_units["CHARGE"]))
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

    bonds = [(b[0].index, b[1].index) for b in topology.bonds()]

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
        bonds=bonds
    )

    return GrappaForceField(xml_string)

# %%
ff = create_forcefield(top, params)
# %%
system = ff.createSystem(top)
# %%
for i, atom in enumerate(system.atoms()):
    print(atom)
    break

#%%


def get_single_res_top(topology):
    topo = Topology()
    chain = topo.addChain()
    res = topo.addResidue("XXX", chain)

    atom_map = {}
    atoms = list(topology.atoms())

    for i, atom in enumerate(atoms):
        new_atom = topo.addAtom(str(i), atom.element, res)
        atom_map[atom] = new_atom

    for a1, a2 in topology.bonds():
        topo.addBond(atom_map[a1], atom_map[a2])

    return topo


orig_top = pdbfile.topology
topo = get_single_res_top(orig_top)
# %%
for res in orig_top.residues():
    print(f"Residue {res.index}: {res.name}")
    print("  Atoms:")
    for atom in res.atoms():
        print(f"    {atom.name} ({atom.element.symbol})")
    print("  Bonds:")
    for bond in res.bonds():
        print(f"    {bond[0].name} - {bond[1].name}")

# %%

for res in topo.residues():
    print(f"Residue {res.index}: {res.name}")
    print("  Atoms:")
    for atom in res.atoms():
        print(f"    {atom.name} ({atom.element.symbol})")
    print("  Bonds:")
    for bond in res.bonds():
        print(f"    {bond[0].name} - {bond[1].name}")
# %%
