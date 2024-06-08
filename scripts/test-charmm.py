from openmm.app import ForceField, PDBFile
from pathlib import Path


# Load the CHARMM force field
ffpath = Path(__file__).parent.parent / 'src' / 'grappa' / 'utils' / 'classical_forcefields' / 'charmm36-jul2022.xml'
ff = ForceField(ffpath)

# Load the PDB file
pdb = PDBFile('tripep.pdb')

# Create the OpenMM system
ff.createSystem(pdb.topology)