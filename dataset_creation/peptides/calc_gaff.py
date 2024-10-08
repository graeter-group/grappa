from grappa.data import MolData
from pathlib import Path
from tqdm import tqdm
from grappa.utils import openmm_utils, openff_utils
import numpy as np
from grappa.utils import get_data_path
import matplotlib.pyplot as plt
import numpy as np
from grappa.utils.plotting import scatter_plot


def reparametrize_dataset(dspath:Path):
    """
    Reparametrize the dataset dspath/*.npz using the force field force_field.
    """
    all_gradients_new = []
    all_gradients_old = []

    err = None
    errs = 0

    assert dspath.is_dir()
    mol_data_paths = list(dspath.glob("*.npz"))
    assert len(mol_data_paths) > 0

    ff_name = "gaff-2.11"
    forcefield = "gaff-2.11"

    for path in tqdm(list(mol_data_paths), desc="Reparametrizing"):

        old_moldata = MolData.load(path)
        pdbstring = old_moldata.pdb
        mapped_smiles = old_moldata.mapped_smiles
        if ff_name in old_moldata.ff_gradient:
            continue

        try:
            system, topology, _ = openff_utils.get_openmm_system(mapped_smiles=mapped_smiles, openff_forcefield=forcefield, partial_charges=old_moldata.molecule.partial_charges)

            energy, forces = openmm_utils.get_energies(system, old_moldata.xyz)
        except Exception as e:
            print(f"Error in {path}")
            errs += 1
            err = e
            continue

        gradients = -forces

        old_moldata.ff_gradient[ff_name] = {"total": gradients}
        old_moldata.ff_energy[ff_name] = {"total": energy}

        old_moldata.save(path=path)

        all_gradients_new.append(old_moldata.ff_gradient[ff_name]['total'].flatten())
        all_gradients_old.append(old_moldata.ff_gradient['openff-1.2.0']['total'].flatten())

    if err is not None:
        print(f"{errs} errors occured for {dspath.stem}")
        raise err

    # plot the different contributions:
    fig, ax = plt.subplots(1,1, figsize=(5,5))

    grads_old = np.concatenate(all_gradients_old)
    grads_new = np.concatenate(all_gradients_new)

    scatter_plot(ax, grads_old, grads_new, cluster=True, logscale=True, show_rmsd=True)

    ax.set_xlabel("openff-1.2.0")
    ax.set_ylabel(ff_name)

    plt.tight_layout()

    plt.savefig(f"gaff-{dspath.stem}.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dspath", type=str, help="Path to the dataset")

    args = parser.parse_args()

    if not args.dspath.startswith("/"):
        args.dspath = get_data_path()/"datasets"/args.dspath

    assert args.dspath.is_dir()

    reparametrize_dataset(Path(args.dspath))