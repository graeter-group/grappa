import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def validate_qm_data(folder):
    # Collect all energies
    psi4_energies = []
    openmm_energies = []
    psi4_forces = []
    openmm_forces = []

    for pdb_folder in Path(folder).iterdir():
        if pdb_folder.is_dir():
            try:
                psi4_energy = np.load(str(pdb_folder/Path("psi4_energies.npy")))
                openmm_energy = np.load(str(pdb_folder/Path("openmm_energies.npy")))
                psi4_force = np.load(str(pdb_folder/Path("psi4_forces.npy")))
                openmm_force = np.load(str(pdb_folder/Path("openmm_forces.npy")))
                psi4_energies.append(psi4_energy)
                openmm_energies.append(openmm_energy)
                psi4_forces.append(psi4_force)
                openmm_forces.append(openmm_force)
            except:
                print(f"failed to load data for {pdb_folder}")
                pass

    # Convert lists to numpy arrays
    psi4_energies = np.concatenate(psi4_energies)
    openmm_energies = np.concatenate(openmm_energies)
    psi4_forces = np.concatenate(psi4_forces)
    openmm_forces = np.concatenate(openmm_forces)

    # Subtract the mean of the energy per molecule
    psi4_energies -= np.mean(psi4_energies)
    openmm_energies -= np.mean(openmm_energies)

    # Plot the data
    fig, ax = plt.subplots(2, 1)

    ax[0].scatter(psi4_energies, openmm_energies)
    ax[0].set_xlabel('psi4 energy')
    ax[0].set_ylabel('openmm energy')

    ax[1].scatter(psi4_forces, openmm_forces)
    ax[1].set_xlabel('psi4 force')
    ax[1].set_ylabel('openmm force')

    plt.tight_layout()
    plt.savefig('summary.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate QM data for a given folder.')
    parser.add_argument('--folder', type=str, help='The folder containing the PDB files.', default="pep3")
    args = parser.parse_args()
    validate_qm_data(args.folder)