VERSION = "1"
N_STATES = 10

from grappa.PDBData.PDBDataset import PDBDataset

from grappa.constants import DEFAULT_DIPEP_PATH
from pathlib import Path


ds = PDBDataset.from_spice(DEFAULT_DIPEP_PATH, n_max=1)

atomic_numbers = ds[0].elements
positions = ds[0].xyz[:N_STATES]
energies = ds[0].energies[:N_STATES]
gradients = ds[0].gradients[:N_STATES]

import numpy as np

dir = Path(__file__).parent/Path(VERSION)
dir.mkdir(exist_ok=True)
np.save(str(dir/Path("atomic_numbers.npy")), atomic_numbers)
np.save(str(dir/Path("positions.npy")), positions)
np.save(str(dir/Path("spice_energies.npy")), energies)
np.save(str(dir/Path("spice_gradients.npy")), gradients)