
import numpy as np

from typing import Tuple, Set, Dict, Union, List


def get_sp_hybridization_encoding(openff_mol:"openff.toolkit.Molecule")->np.ndarray:
    """
    Returns a numpy array of shape (n_atoms, 6) that one-hot encodes wheter the atom can be described by a given hybridization type.
    """
    from rdkit.Chem.rdchem import HybridizationType

    # define the one hot encoding:
    hybridization_conversion = [
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
        HybridizationType.S,
    ]

    mol = openff_mol.to_rdkit()
    return np.array(
            [
                [
                    int(atom.GetHybridization() == hybridization) for hybridization in hybridization_conversion
                ]
                for atom in mol.GetAtoms()
            ]
        )