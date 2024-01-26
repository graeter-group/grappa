"""
Since we want to be independent from openff and the hybridization feature turns out to be helpful, write something similar by hand.
"""

#%%
from openff.toolkit.topology import Molecule as OFFMolecule

# some molecule with C, O, N, H with explicit hydrogens:
smiles = "[H][O][P](=[O])([O][H])[C@]([H])([N]([H])[C]([H])([H])[c]1[c]([H])[c]([C]#[N])[c]([H])[c]2[c]1[N]([H])[C](=[O])[C](=[O])[N]2[H])[C]([H])([H])[H]"

openff_mol = OFFMolecule.from_smiles(smiles)
openff_mol
# %%
# calculate the hybridization with openff/rdkit:

mol_rd = openff_mol.to_rdkit()

hybridizations_ = [
    atom.GetHybridization() for atom in mol_rd.GetAtoms()
]
hybridizations_
# %%
from grappa.data import Molecule
import numpy as np

mol_ = Molecule.from_openff_molecule(openff_mol, partial_charges=np.zeros(openff_mol.n_atoms))
mol_.add_features("sp_hybridization")
hybridizations = mol_.additional_features["sp_hybridization"]

# %%
# invert onehot encoding:
from grappa.utils.hybridization import HybridizationState

states = list(HybridizationState)

hybridizations = [states[vector.argmax()] for vector in hybridizations]
hybridizations

# %%
elements = [mol_rd.GetAtomWithIdx(i).GetSymbol() for i in range(openff_mol.n_atoms)]

# print all alongside:
for i, (hybridization, hybridization_, el) in enumerate(zip(hybridizations, hybridizations_, elements)):
    print(f"{i}: {hybridization} {hybridization_} {el}")
# %%
def convert_hybridization(rd_hybridization):
    from rdkit.Chem.rdchem import HybridizationType
    if rd_hybridization == HybridizationType.SP:
        return HybridizationState.SP
    elif rd_hybridization == HybridizationType.SP2:
        return HybridizationState.SP2
    elif rd_hybridization == HybridizationType.SP3:
        return HybridizationState.SP3
    elif rd_hybridization == HybridizationType.SP3D:
        return HybridizationState.SP3D
    elif rd_hybridization == HybridizationType.SP3D2:
        return HybridizationState.SP3D2
    elif rd_hybridization == HybridizationType.S:
        return HybridizationState.S
    else:
        return HybridizationState.UNKNOWN

def test_same_hybridizations(openff_mol):
    rd_mol = openff_mol.to_rdkit()
    rd_hybridizations = [atom.GetHybridization() for atom in rd_mol.GetAtoms()]
    mol = Molecule.from_openff_molecule(openff_mol, partial_charges=np.zeros(openff_mol.n_atoms))
    mol.add_features("sp_hybridization")
    hybridizations = mol.additional_features["sp_hybridization"]
    # invert the one-hot encoding:
    states = list(HybridizationState)
    hybridizations = [states[vector.argmax()] for vector in hybridizations]

    # for all that are not unknown, check that they are the same:
    for i, (hybridization, rd_hybridization) in enumerate(zip(hybridizations, rd_hybridizations)):
        if hybridization != HybridizationState.UNKNOWN:
            if not hybridization == convert_hybridization(rd_hybridization):
                raise RuntimeError(f"Hybridization {i} is not the same: hybridization={hybridization}, rd_hybridization={rd_hybridization}, element={rd_mol.GetAtomWithIdx(i).GetSymbol()}, neighbors: {len(mol.neighbor_dict[mol.atoms[i]])}")

test_same_hybridizations(openff_mol)
# %%
from pathlib import Path
import json

splitpath = str(Path(__file__).parent.parent.parent/f"dataset_creation/get_espaloma_split/espaloma_split.json")

with open(splitpath, "r") as f:
    split = json.load(f)
# %%
all_smiles = split['train'] + split['val'] + split['test']
print(len(all_smiles))
# %%
for i, smilestring in enumerate(all_smiles[:100]):
    print(i, end="\r")
    openff_mol = OFFMolecule.from_smiles(smilestring, allow_undefined_stereo=True)
    test_same_hybridizations(openff_mol)
# %%
openff_mol
# %%
