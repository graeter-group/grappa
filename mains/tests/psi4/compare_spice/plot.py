#%%
def _plot():
    VERSION = "1"
    import numpy as np
    import matplotlib.pyplot as plt

    from pathlib import Path

    dir = Path(__file__).parent/Path(VERSION)

    # Load the energies and forces:

    psi4_energies = np.load(str(dir/Path("psi4_energies.npy"))).flatten()
    psi4_forces = np.load(str(dir/Path("psi4_forces.npy"))).flatten()
    spice_energies = np.load(str(dir/Path("spice_energies.npy")))[:len(psi4_energies)].flatten()
    spice_forces = -np.load(str(dir/Path("spice_gradients.npy")))[:len(psi4_energies)].flatten()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Scatter plots
    axs[0].scatter(psi4_energies, spice_energies)
    axs[0].plot(spice_energies, spice_energies, color='orange')
    axs[0].set_xlabel('Psi4 energies (kcal/mol)')
    axs[0].set_ylabel('Spice energies (kcal/mol)')
    axs[0].set_title('Energies comparison')

    axs[1].scatter(psi4_forces, spice_forces)
    axs[1].plot(spice_forces, spice_forces, color='orange')
    axs[1].set_xlabel('Psi4 forces (kcal/mol/angstrom)')
    axs[1].set_ylabel('Spice forces (kcal/mol/angstrom)')
    axs[1].set_title('Forces comparison')

    plt.tight_layout()
    plt.savefig(str(dir/Path('eval.png')))

    