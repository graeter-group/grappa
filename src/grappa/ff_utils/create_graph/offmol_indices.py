# inspired from espaloma:


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


import numpy as np

# supress openff warning:
import logging
logging.getLogger("openff").setLevel(logging.ERROR)
from openff.toolkit.topology import Molecule


def atom_indices(offmol: Molecule) -> np.ndarray:
    return np.array([a.molecule_atom_index for a in offmol.atoms])


def bond_indices(offmol: Molecule) -> np.ndarray:
    return np.array([(b.atom1_index, b.atom2_index) for b in offmol.bonds])


def angle_indices(offmol: Molecule) -> np.ndarray:
    return np.array(
        sorted(
            [
                tuple([atom.molecule_atom_index for atom in angle])
                for angle in offmol.angles
            ]
        )
    )


def proper_torsion_indices(offmol: Molecule) -> np.ndarray:
    return np.array(
        sorted(
            [
                tuple([atom.molecule_atom_index for atom in proper])
                for proper in offmol.propers
            ]
        )
    )


def _all_improper_torsion_indices(offmol: Molecule) -> np.ndarray:
    """"[*:1]~[*:2](~[*:3])~[*:4]" matches"""

    return np.array(
        sorted(
            [
                tuple([atom.molecule_atom_index for atom in improper])
                for improper in offmol.impropers
            ]
        )
    )


def improper_torsion_indices(offmol: Molecule) -> np.ndarray:
    """
    Returns the indices of all improper torsions (i.e. without dividing out the symmetry!) in the molecule such that the central atom is at index 2, i.e. in the third position, in accordance to amber.
    """

    ## Find all atoms bound to exactly 3 other atoms

    ## This finds all orderings, which is what we want for the espaloma case
    ##  but not for smirnoff
    improper_smarts = '[*:1]~[X3:2](~[*:3])~[*:4]'
    mol_idxs = offmol.chemical_environment_matches(improper_smarts)
    mol_idxs = np.array(mol_idxs)
    # now the central atom is at index 1, therefore we simply permute 1 and 2:
    mol_idxs = mol_idxs[:, [0, 2, 1, 3]]
    return mol_idxs
