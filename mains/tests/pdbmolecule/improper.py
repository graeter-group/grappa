#%%
from grappa.PDBData.PDBMolecule import PDBMolecule
m = PDBMolecule.from_pdb("my_pdb.pdb")
# %%
from openmm.app import ForceField
m.to_dgl(classical_ff=ForceField("amber99sbildn.xml"))
# %%

import rdkit
from typing import Tuple
from grappa.ff_utils.create_graph import utils


def is_improper_(rd_mol:rdkit.Chem.rdchem.Mol, idxs:Tuple[int,int,int,int], central_atom_idx:int=2)->bool:
    """
    Helper function to check whether the given tuple of indices describes an improper torsion.
    Checks whether the given tuple of indices describes an improper torsion.
    We can assume that the tuples describe either a proper or improper torsion.
    We also assume that the idxs correspond to the indices of the rdkit molecule.
    """
    # check whether the central atom is the connected to all other atoms in the rdkit molecule.

    central_atom = rd_mol.GetAtomWithIdx(idxs[central_atom_idx])

    # get the neighbors of the central atom
    neighbor_idxs = set([n.GetIdx() for n in central_atom.GetNeighbors()])

    # for each atom in the torsion, check if it's a neighbor of the central atom
    for i, atom_idx in enumerate(idxs):
        if i != central_atom_idx:  # skip the central atom itself
            if atom_idx not in neighbor_idxs:
                # if one of the atoms is not connected to it, return False
                return False

    # if all atoms are connected to the central atom, this is an improper torsion
    return True


# create an rdkit molecule
rd_mol = utils.openmm2rdkit_graph(openmm_top=m.to_openmm().topology)
# %%
idxs = (3,4,2,5)
is_improper_(rd_mol, idxs)
# %%
