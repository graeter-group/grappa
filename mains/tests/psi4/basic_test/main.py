from ase import Atoms
from ase.calculators.psi4 import Psi4
from ase.io import read, write
from ase import units as ase_units
import numpy as np
import matplotlib.pyplot as plt

# import openmm as mm
from pathlib import Path



def calc_states(pdb_folder=Path(__file__).parent, n_states=None):

    pdb_folder = Path(pdb_folder)

    if not (pdb_folder/Path("positions.npy")).exists():
        raise FileNotFoundError(f"positions.npy not found in {pdb_folder}")

    positions = np.load(str(pdb_folder/Path("positions.npy")))
    atomic_numbers = np.load(str(pdb_folder/Path("atomic_numbers.npy")))

  
    # Calculate energies and forces using Psi4
    psi4_energies = []
    psi4_forces = []


    from time import time
    start = time()

    missing_indices = range(len(psi4_energies), len(positions))

    # restroct the amount of calculations:
    if n_states is not None:
        if len(missing_indices) > n_states:
            missing_indices = missing_indices[:n_states]

    for num_calculated, i in enumerate(missing_indices):

        msg = f"calculating state number {i}/{len(positions)-1}... Progress: {num_calculated}/{len(missing_indices)-1}, time elapsed: {round((time() - start)/60., 2)} min"
        if i > 0:
            msg += f", avg time per state: {round((time() - start)/(i) / 60.,2)} min"
        print(msg)

        # Read the configuration
        atoms = Atoms(numbers=atomic_numbers, positions=positions[i])

        atoms.set_calculator(Psi4(atoms = atoms, method = 'bmk', memory = '20GB', basis = '6-311+G(2df,p)', num_threads=10))
        energy = atoms.get_potential_energy(apply_constraint=False)
        forces = atoms.get_forces(apply_constraint=False)

        EV_IN_KCAL = 23.0609
        # EV = mm.unit.kilocalorie_per_mole * EV_IN_KCAL
        # energy = mm.unit.Quantity(energy, EV).value_in_unit(mm.unit.kilocalories_per_mole)
        # forces = mm.unit.Quantity(forces, EV/mm.unit.angstrom).value_in_unit(mm.unit.kilocalories_per_mole/mm.unit.angstrom)

        energy = energy * EV_IN_KCAL
        forces = forces * EV_IN_KCAL



        psi4_energies.append(energy)
        psi4_forces.append(forces)

        # save the energies and forces in every step:
        np.save(str(pdb_folder/Path("psi4_energies.npy")), np.array(psi4_energies))
        np.save(str(pdb_folder/Path("psi4_forces.npy")), np.array(psi4_forces))

def validate_qm_data():
    # Collect all energies
    psi4_energies = []
    openmm_energies = []
    psi4_forces = []
    openmm_forces = []

    pdb_folder = Path(__file__).parent

    psi4_energy = np.load(str(pdb_folder/Path("psi4_energies.npy")))
    openmm_energy = np.load(str(pdb_folder/Path("openmm_energies.npy")))
    psi4_force = np.load(str(pdb_folder/Path("psi4_forces.npy")))
    openmm_force = np.load(str(pdb_folder/Path("openmm_forces.npy")))

    if len(psi4_energy) != len(psi4_force):
        raise Exception(f"psi4 energy and force have different lengths for {pdb_folder}")
    if len(openmm_energy) != len(openmm_force):
        raise Exception(f"openmm energy and force have different lengths for {pdb_folder}")

    openmm_energy = openmm_energy[:len(psi4_energy)]
    openmm_force = openmm_force[:len(psi4_force)]
    

    # Subtract the mean of the energy per molecule
    psi4_energy -= np.mean(psi4_energy)
    openmm_energy -= np.mean(openmm_energy)


    psi4_energies.append(psi4_energy.flatten())
    openmm_energies.append(openmm_energy.flatten())
    psi4_forces.append(psi4_force.flatten())
    openmm_forces.append(openmm_force.flatten())


    # Convert lists to numpy arrays
    psi4_energies = np.concatenate(psi4_energies)
    openmm_energies = np.concatenate(openmm_energies)
    psi4_forces = np.concatenate(psi4_forces)
    openmm_forces = np.concatenate(openmm_forces)


    # Plot the data
    fig, ax = plt.subplots(1,2, figsize=(10,5))

    ax[0].scatter(psi4_energies, openmm_energies)
    ax[0].plot(psi4_energies, psi4_energies, color='black')
    ax[0].set_xlabel('psi4 energy [kcal/mol]')
    ax[0].set_ylabel('openmm energy [kcal/mol]')

    ax[1].scatter(psi4_forces, openmm_forces)
    ax[1].plot(psi4_forces, psi4_forces, color='black')
    ax[1].set_xlabel('psi4 force [kcal/mol/Å]')
    ax[1].set_ylabel('openmm force [kcal/mol/Å]')


    rmse_energies = np.sqrt(np.mean((psi4_energies - openmm_energies)**2))

    rmse_forces = np.sqrt(np.mean((psi4_forces - openmm_forces)**2))

    ax[0].text(0.05, 0.95, f"RMSE: {rmse_energies:.2f} kcal/mol", transform=ax[0].transAxes, verticalalignment='top')


    ax[1].text(0.05, 0.95, f"RMSE: {rmse_forces:.2f} kcal/mol/Å", transform=ax[1].transAxes, verticalalignment='top')

    plt.tight_layout()

    plt.savefig(f'summary.png')

calc_states(n_states=2)
validate_qm_data()
