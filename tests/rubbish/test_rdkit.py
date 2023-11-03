#%%
from rdkit.Chem.rdchem import Mol
from rdkit import Chem
from rdkit.Chem import rdchem
#%%

bonds = [(0, 1), (1, 2), (2, 3), (3, 4)]
atoms = [0, 1, 2, 3, 4]

def from_rdkit(atoms, bonds):
    mol = Chem.RWMol()

    for atom in atoms:
        mol.AddAtom(rdchem.Atom(0))
    for a1, a2 in bonds:
        mol.AddBond(a1, a2, rdchem.BondType.SINGLE)
    mol = mol.GetMol()



#%%


#%%