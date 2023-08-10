from pathlib import Path

DEFAULTBASEPATH = "/hits/fast/mbm/seutelf/data/datasets/PDBDatasets"

SPICEPATH = "/hits/fast/mbm/seutelf/data/datasets/SPICE-1.1.2.hdf5"
DEFAULT_DIPEP_PATH = str(Path(SPICEPATH).parent/Path("dipeptides_spice.hdf5"))

# MAX_ELEMENT = 26 # cover Iron

MAX_ELEMENT = 53 # cover Iodine

RESIDUES = ['ACE', 'NME', 'CYS', 'ASP', 'SER', 'GLN', 'LYS', 'ILE', 'PRO', 'THR', 'PHE', 'ASN', 'GLY', 'HIS', 'LEU', 'ARG', 'TRP', 'ALA', 'VAL', 'GLU', 'TYR', 'MET', "HIE", "HID", "HIP", "DOP", "HYP"]

ONELETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HIE": "H",
    "HIP": "H",
    "HID": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "HYP": "O",
    "DOP": "J",
    "ACE": "B",
    "NME": "Z"
}

# custom types:


try:
    import numpy as np
    from typing import TypedDict, List, Tuple, Union, Dict, Optional

    class TopologyDict(TypedDict):
        """
        A dictionary with keys:
            - atoms: List[Tuple[...]]
            - bonds: List[Tuple[int,int]]
            - (optionally) radicals: List[int]

        Defines the topology dictionary that can be used to initialize the system.
        The atoms entry is a list of atoms, where each entry (each atom) is of the form:
        (
            index: int,
            pdb-style atom_type: str,
            3-letter-residue: str,
            residue_index: int,
            (
                sigma: float,
                epsilon: float
            )
            atomic_number: int,
        )
        The atom indices must not be ordered or zero-based. The residue_indices must be grouped together but neither increasing not zero-based.
        e.g.
            [42, "CH3", "ACE", 2, [0.339967, 0.45773], 6]
            [7, "C", "ACE", 2, [0.339967, 0.45773], 6]

        Sigma and epsilon are the Lennard-Jones parameters of the atom. They must not be specified and can be set to None. In this case the parameters will be calculated using the classical force field.

        Optionally, a list keyed 'radicals' can be given containing the atom indices of radical atoms in the topology.
        """
        atoms: List[Tuple[int, str, str, int, Tuple[float, float], int]]
        bonds: List[Tuple[int, int]]

    class ParamDict(TypedDict):
        """
        A parameter dict containing index tuples (corresponding to the atom_idx passed in the atoms list) and np.ndarrays:
        
        {
        "atom_idxs":np.array, the indices of the atoms in the molecule that correspond to the parameters. In rising order and starts at zero.

        "atom_q":np.array, the partial charges of the atoms.

        "atom_sigma":np.array, the sigma parameters of the atoms.

        "atom_epsilon":np.array, the epsilon parameters of the atoms.

        
        "{bond/angle}_idxs":np.array of shape (#2/3-body-terms, 2/3), the indices of the atoms in the molecule that correspond to the parameters. The permutation symmetry of the n-body term is already divided out, i.e. this is the minimal set of parameters needed to describe the interaction.

        "{bond/angle}_k":np.array, the force constant of the interaction.

        "{bond/angle}_eq":np.array, the equilibrium distance of the interaction.   

        
        "{proper/improper}_idxs":np.array of shape (#4-body-terms, 4), the indices of the atoms in the molecule that correspond to the parameters. The central atom is at third position, i.e. index 2. For each entral atom, the array contains all cyclic permutation of the other atoms, i.e. 3 entries that all have different parameters in such a way that the total energy is invariant under cyclic permutation of the atoms.

        "{proper/improper}_ks":np.array of shape (#4-body-terms, n_periodicity), the fourier coefficients for the cos terms of torsion. may be negative instead of the equilibrium dihedral angle (which is always set to zero). n_periodicity is a hyperparemter of the model and defaults to 6.

        "{proper/improper}_ns":np.array of shape (#4-body-terms, n_periodicity), the periodicities of the cos terms of torsion. n_periodicity is a hyperparemter of the model and defaults to 6.

        "{proper/improper}_phases":np.array of shape (#4-body-terms, n_periodicity), the phases of the cos terms of torsion. n_periodicity is a hyperparameter of the model and defaults to 6.

        }
        """
        atom_idxs: np.ndarray
        atom_q: np.ndarray
        atom_sigma: np.ndarray
        atom_epsilon: np.ndarray
        bond_idxs: np.ndarray
        bond_k: np.ndarray
        bond_eq: np.ndarray
        angle_idxs: np.ndarray
        angle_k: np.ndarray
        angle_eq: np.ndarray
        proper_idxs: np.ndarray
        proper_ks: np.ndarray
        proper_ns: np.ndarray
        proper_phases: np.ndarray
        improper_idxs: np.ndarray
        improper_ks: np.ndarray
        improper_ns: np.ndarray
        improper_phases: np.ndarray

except ImportError:
    from typing import Dict

    TopologyDict = Dict
    ParamDict = Dict
