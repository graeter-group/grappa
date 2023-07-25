from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdchem import HybridizationType

import torch

def get_chemical_features(mol:Mol) -> torch.Tensor:
    """
    Returns a torch tensor containing information on the hybridization of each atom in the molecule. This should only be called when the mol is constructed with chemical information, e.g. via an openff molecule.
    Output shape: (n_atoms, 5)
    """
    def get_chemical_feature(atom):
        return get_chemical_features.hybridization_conversion[atom.GetHybridization()]

    return torch.stack([get_chemical_feature(atom) for atom in mol.GetAtoms()], dim=0)





# from espaloma:

# MIT License

# Copyright (c) 2020 Yuanqing Wang @ choderalab // MSKCC

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

get_chemical_features.hybridization_conversion = {
    HybridizationType.SP: torch.tensor(
        [1, 0, 0, 0, 0],
        dtype=torch.float32,
    ),
    HybridizationType.SP2: torch.tensor(
        [0, 1, 0, 0, 0],
        dtype=torch.float32,
    ),
    HybridizationType.SP3: torch.tensor(
        [0, 0, 1, 0, 0],
        dtype=torch.float32,
    ),
    HybridizationType.SP3D: torch.tensor(
        [0, 0, 0, 1, 0],
        dtype=torch.float32,
    ),
    HybridizationType.SP3D2: torch.tensor(
        [0, 0, 0, 0, 1],
        dtype=torch.float32,
    ),
    HybridizationType.S: torch.tensor(
        [0, 0, 0, 0, 0],
        dtype=torch.float32,
    ),
}