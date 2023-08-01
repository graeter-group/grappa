import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

def validate_qm_data(folder):
    # Collect all energies
    psi4_energies = []
    openmm_energies = []
    psi4_forces = []
    openmm_forces = []

    for pdb_folder in Path(folder).iterdir():
        if pdb_folder.is_dir():
            if (pdb_folder/Path("psi4_energies.npy")).exists() and (pdb_folder/Path("psi4_forces.npy")).exists():
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
                
                if openmm_energy.shape != psi4_energy.shape:
                    raise Exception(f"psi4 and openmm energies have different shapes for {pdb_folder}: \n{openmm_energy.shape} vs {psi4_energy.shape}")
                if openmm_force.shape != psi4_force.shape:
                    raise Exception(f"psi4 and openmm forces have different shapes for {pdb_folder}: \n{openmm_force.shape} vs {psi4_force.shape}")

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


    plt.savefig(os.path.join(str(folder),"summary.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate QM data for a given folder.')
    parser.add_argument('folder', type=str, help='Folder containing the QM data.')
    args = parser.parse_args()
    validate_qm_data(args.folder)