#%%
from openff.toolkit.topology import Molecule

#%%

smiles = "C1=CC=C(C=C1)C(=O)O"

mol = Molecule.from_smiles(smiles)

mapped_smiles = mol.to_smiles(mapped=True)

#%%
from openff.units import unit

mol.partial_charges = [0] * len(mol.atoms) * unit.elementary_charge
(mol.partial_charges/unit.elementary_charge).magnitude
#%%
atomic_numbers = [atom.atomic_number for atom in mol.atoms]
atomic_numbers
#%%
[mol.atoms[i].molecule_atom_index for i in range(len(mol.atoms))]
#%%
[(bond.atom1_index, bond.atom2_index) for bond in mol.bonds]
# %%
impropers = set([])
for atoms in mol.smirnoff_impropers:
    impropers.add(tuple(sorted((atoms[0]._molecule_atom_index, atoms[1]._molecule_atom_index, atoms[2]._molecule_atom_index, atoms[3]._molecule_atom_index))))
len(impropers)
# %%
from grappa.data import Molecule

mol_grappa = Molecule.from_smiles(mapped_smiles=mapped_smiles, partial_charges=0)
len(mol_grappa.impropers)
# %%
grappa_impropers = set([])
for improper in mol_grappa.impropers:
    grappa_impropers.add(tuple(sorted(improper)))
# %%
assert grappa_impropers == impropers
# %%
mol_grappa.bonds
# %%
mol_grappa2 = Molecule.from_openff_molecule(mol)
# %%
assert set(mol_grappa2.bonds) == set(mol_grappa.bonds)
assert set(mol_grappa2.angles) == set(mol_grappa.angles)
assert set(mol_grappa2.propers) == set(mol_grappa.propers)
assert set(mol_grappa2.impropers) == set(mol_grappa.impropers)
# %%
assert len(mol_grappa2.impropers) == len(mol_grappa.impropers)
assert len(mol_grappa2.propers) == len(mol_grappa.propers)
assert len(mol_grappa2.angles) == len(mol_grappa.angles)
assert len(mol_grappa2.bonds) == len(mol_grappa.bonds)
# %%
