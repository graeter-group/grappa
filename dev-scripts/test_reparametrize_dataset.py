#%%
import numpy as np
from grappa.utils.data_utils import get_moldata_path
from grappa.utils.openmm_utils import get_pdb
from openmm.app import ForceField
from grappa.data import MolData, Molecule, Parameters
from grappa.utils.openmm_utils import get_single_res_top, GrappaForceField

paths = list(get_moldata_path("spice-dipeptide-amber99").glob("*.npz"))
paths = sorted(paths, key=lambda x: x.name)
moldata = MolData.load(paths[20])
num_impropers = len(moldata.molecule.impropers)
print(f"Number of impropers: {num_impropers}")
pdbfile = get_pdb(moldata.pdb)
num_atoms = len(list(pdbfile.topology.atoms()))
print(f"Number of atoms in PDB: {num_atoms}")
top = pdbfile.topology

molecule = moldata.molecule
params = moldata.classical_parameters

partial_charges = molecule.partial_charges

params.partial_charges = molecule.partial_charges
params.epsilons = np.ones(len(partial_charges)) * 0.1  # Set a default epsilon value
params.sigmas = np.ones(len(partial_charges)) * 0.1  # Set a default sigma value

forcefield = GrappaForceField.from_parameters(params, top)
system = forcefield.createSystem(top)

moldata_new = MolData.from_openmm_system(
    openmm_system=system,
    openmm_topology=pdbfile.topology,
    xyz=moldata.xyz,
    energy=moldata.energy,
    gradient=moldata.gradient,
    mol_id=moldata.mol_id,
    mapped_smiles=moldata.mapped_smiles,
    pdb=moldata.pdb,
    ff_name="new_nonbonded",
    sequence=moldata.sequence,
    smiles=moldata.smiles
)

#%%

energy_angle = moldata.ff_energy['amber99sbildn']['angle']
energy_angle_new = moldata_new.ff_energy['new_nonbonded']['angle']

np.max(np.abs(energy_angle - energy_angle_new))
# %%

energy_nonbonded = moldata.ff_energy['amber99sbildn']['nonbonded']
energy_nonbonded_new = moldata_new.ff_energy['new_nonbonded']['nonbonded']
np.max(np.abs(energy_nonbonded - energy_nonbonded_new))

# %%
