VERSION = "2"
from ase import Atoms
from ase.calculators.psi4 import Psi4
from ase.io import read, write
from ase import units as ase_units
import numpy as np

import openmm as mm
from pathlib import Path

dir = Path(__file__).parent/Path(VERSION)

positions = np.load(str(dir/Path("positions.npy")))
atomic_numbers = np.load(str(dir/Path("atomic_numbers.npy")))


# Calculate energies and forces using Psi4
psi4_energies = []
psi4_forces = []

if (dir/Path("psi4_energies.npy")).exists():
    psi4_energies = np.load(str(dir/Path("psi4_energies.npy"))).tolist()
    psi4_forces = np.load(str(dir/Path("psi4_forces.npy"))).tolist()

from time import time
start = time()

for i in range(len(psi4_energies), len(positions)):
    if i > 1:
        from plot import _plot
        _plot()

    msg = f"calculating {i}..., time elapsed: {time() - start}"
    if i > 0 and not (dir/Path("psi4_energies.npy")).exists():
        msg += f", avg time per state: {(time() - start)/(i)}"
    print(msg)
    # Read the configuration
    atoms = Atoms(numbers=atomic_numbers, positions=positions[i])
    calc = Psi4(atoms = atoms, method = 'bmk', memory = '20GB', basis = '6-311+G(2df,p)', num_threads=18)

    energy = atoms.get_potential_energy(apply_constraint=False)
    forces = atoms.get_forces(apply_constraint=False).flatten()

    EV = mm.unit.kilocalorie_per_mole * 23.0609
    energy = mm.unit.Quantity(energy, EV).value_in_unit(mm.unit.kilocalories_per_mole)
    forces = mm.unit.Quantity(forces, EV/mm.unit.angstrom).value_in_unit(mm.unit.kilocalories_per_mole/mm.unit.angstrom)

    psi4_energies.append(energy)
    psi4_forces.append(forces)

    # save the current energies and forces:
    np.save(str(dir/Path("psi4_energies.npy")), np.array(psi4_energies))
    np.save(str(dir/Path("psi4_forces.npy")), np.array(psi4_forces))

from plot import _plot
_plot()