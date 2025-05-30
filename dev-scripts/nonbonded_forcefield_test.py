#%%
from grappa.utils.data_utils import get_moldata_path
from grappa.utils.openmm_utils import get_pdb
from openmm.app import ForceField
from grappa.data import MolData, Molecule, Parameters

paths = list(get_moldata_path("spice-dipeptide-amber99").glob("*.npz"))
paths = sorted(paths, key=lambda x: x.name)
moldata = MolData.load(paths[20])
num_impropers = len(moldata.molecule.impropers)
print(f"Number of impropers: {num_impropers}")
pdbfile = get_pdb(moldata.pdb)
num_atoms = len(list(pdbfile.topology.atoms()))
print(f"Number of atoms in PDB: {num_atoms}")
forcefield = ForceField("amber99sb.xml")
system = forcefield.createSystem(pdbfile.topology)
top = pdbfile.topology


molecule = Molecule.from_openmm_system(openmm_system=system, openmm_topology=pdbfile.topology)
params = Parameters.from_openmm_system(openmm_system=system, mol=molecule)

assert list(range(len(params.atoms))) == params.atoms
top = pdbfile.topology
for i, atom in enumerate(top.atoms()):
    mass = atom.element.mass
    symbol = atom.element.symbol
    # print(mass)


from grappa.utils.openmm_utils import get_single_res_top, GrappaForceField

# %%
ff = GrappaForceField.from_parameters(params, top)
top = pdbfile.topology

system_new = ff.createSystem(top)

top = pdbfile.topology

topo = get_single_res_top(top)
for res in top.residues():
    print(res.name, res.index)

print()

for res in topo.residues():
    print(res.name, res.index)

for force in system_new.getForces():
    print(force.__class__.__name__)
    # if 'torsion' in force.__class__.__name__.lower():
    #     for i in range(force.getNumTorsions()):
    #         torsion = force.getTorsionParameters(i)
    #         print(torsion)

#%%

from grappa.utils.openmm_utils import get_energies, remove_forces_from_system
import matplotlib.pyplot as plt

xyz = moldata.xyz

energies_ff, forces_ff = get_energies(openmm_system=system, xyz=xyz)

energies_old, forces_old = get_energies(openmm_system=system_new, xyz=xyz)

plt.scatter(energies_ff, energies_old)
plt.show()
plt.scatter(forces_ff, forces_old)
# %%
system = remove_forces_from_system(system, keep=["torsion"])
system_new = remove_forces_from_system(system_new, keep=["torsion"])

xyz = moldata.xyz

energies_ff, forces_ff = get_energies(openmm_system=system, xyz=xyz)

energies_old, forces_old = get_energies(openmm_system=system_new, xyz=xyz)

plt.scatter(energies_ff, energies_old)
plt.show()
plt.scatter(forces_ff, forces_old)


# %%
