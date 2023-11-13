"""
Create a grappa molecule from a smiles string: Use a small molecule forcefield to parametrize.
"""
#%%
from grappa.data import Molecule, Parameters

# %%
# some smiles string:
smilestring = 'CC(C)(C)C(=O)OC1CC2C3CCC4=CC(=O)CCC4=C3C(CCC12C)C(=O)OC(C)(C)C'

# very small molecule:
# smilestring = 'CC'


from openff.toolkit import ForceField
from openff.toolkit import Molecule as OpenFFMolecule
from openff.units import unit

openff_mol = OpenFFMolecule.from_smiles(smilestring, allow_undefined_stereo=True)

# set charges to zero to make it faster.
openff_mol.partial_charges = [0] * len(openff_mol.atoms) * unit.elementary_charge

ff = ForceField('openff_unconstrained-1.2.0.offxml')
system = ff.create_openmm_system(openff_mol.to_topology(), charge_from_molecules=[openff_mol])
mol = Molecule.from_openmm_system(system, openmm_topology=openff_mol.to_topology().to_openmm())
#%%
params = Parameters.from_openmm_system(system, mol=mol)
# %%
params.bond_eq[:10]
# %%
params.angle_eq[:20]/3.14
# %%
g = mol.to_dgl()
g = params.write_to_dgl(g)
# %%
# now without impropers:
small_openff_mol = OpenFFMolecule.from_smiles('CC', allow_undefined_stereo=True)

mapped_smiles = small_openff_mol.to_smiles(mapped=True)
small_mol = Molecule.from_smiles(mapped_smiles, partial_charges=0)

small_openff_mol.partial_charges = [0] * len(small_openff_mol.atoms) * unit.elementary_charge

system = ff.create_openmm_system(small_openff_mol.to_topology(), charge_from_molecules=[small_openff_mol])

params = Parameters.from_openmm_system(system, mol=small_mol)

#%%
g = small_mol.to_dgl()
g = params.write_to_dgl(g)
# %%
import torch
assert torch.all(g.nodes['n4_improper'].data['k_ref'] == torch.zeros((0,6)))
# %%
