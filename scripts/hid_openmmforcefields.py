#%%
from openmmforcefields.generators import SystemGenerator
from openff.toolkit.topology import Molecule as OFFMol

def get_peptide_system(mol:"openff.Molecule", ff="amber99sbildn.xml")->"openmm.System":
    """
    Assuming that residue information is stored in the openff molecule, returns a parameterized openmm system.
    """

    generator = SystemGenerator(forcefields=[ff], molecules=[mol], forcefield_kwargs={"constraints": None, "removeCMMotion": False},
    )

    return generator.create_system(mol.to_topology().to_openmm())

import openmm
from openmm import app
#%%
# pdbfile = 'capped_HID_renamed.pdb'

# pdb = app.PDBFile(pdbfile)

# # energy minimization
# from openmm import unit
# from openmm.app import Simulation

# system = app.ForceField('amber99sbildn.xml').createSystem(pdb.topology)

# integrator = openmm.VerletIntegrator(0.001*unit.femtoseconds)
# simulation = Simulation(pdb.topology, system, integrator)
# simulation.context.setPositions(pdb.positions)

# simulation.minimizeEnergy()

# pdb.positions = simulation.context.getState(getPositions=True).getPositions()
# app.PDBFile.writeFile(pdb.topology, pdb.positions, open('capped_HID.pdb', 'w'))

# %%
from openmm import unit
mol_HIE = OFFMol.from_polymer_pdb('capped_HIE.pdb')
mol_HID = OFFMol.from_polymer_pdb('capped_HID.pdb')

def get_partial_charges(mol):  
    system = get_peptide_system(mol)

    heavy_atoms = [atom.index for atom in mol_HIE.to_topology().to_openmm().atoms() if atom.element.name != 'hydrogen']

    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            if not nonbonded_force is None:
                raise ValueError("More than one nonbonded force found.")
            nonbonded_force = force

    heavy_atom_partial_charges = [nonbonded_force.getParticleParameters(i)[0] for i in heavy_atoms]
    return heavy_atom_partial_charges
# %%
print([charge.value_in_unit(unit.elementary_charge) for charge in get_partial_charges(mol_HIE)])
print([charge.value_in_unit(unit.elementary_charge) for charge in get_partial_charges(mol_HID)])
# %%
# generate unique isomeric SMILES
mol_HID.to_smiles(mapped=False, isomeric=True)
# %%
