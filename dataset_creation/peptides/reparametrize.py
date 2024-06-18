from grappa.data import MolData
from pathlib import Path
from tqdm import tqdm
from grappa.utils import openmm_utils, openff_utils
import numpy as np
from grappa.utils import get_data_path
import matplotlib.pyplot as plt
import numpy as np
from grappa.utils.plotting import scatter_plot


def reparametrize_dataset(dspath:Path, outpath:Path, forcefield:str, ff_type:str="openmm", charge_model:str="", crmse_limit:float=15., old_ff_name:str="amber99sbildn", charge_noise:float=0., plot_only:bool=False):
    """
    Reparametrize the dataset dspath/*.npz using the force field force_field.
    """
    CONTRIBS = ['bond', 'angle', 'proper', 'improper', 'nonbonded', 'total']

    all_gradients_new = {contrib: [] for contrib in CONTRIBS}
    all_gradients_old = {contrib: [] for contrib in CONTRIBS}

    assert dspath.is_dir()
    mol_data_paths = list(dspath.glob("*.npz"))
    assert len(mol_data_paths) > 0

    ff_name = forcefield
    ff_name = ff_name.replace('.xml', '')
    ff_name = ff_name.replace('.offxml', '')
    ff_name = ff_name.replace('_unconstrained', '')

    for path in tqdm(list(mol_data_paths), desc="Reparametrizing"):

        old_moldata = MolData.load(path)

        if not plot_only:
            pdbstring = old_moldata.pdb
            mapped_smiles = old_moldata.mapped_smiles

            if ff_type == 'openmm':
                # get topology:
                topology = openmm_utils.topology_from_pdb(pdbstring)
                ff = openmm_utils.get_openmm_forcefield(forcefield)
                system = ff.createSystem(topology)

            elif ff_type == 'openff' or ff_type == 'openmmforcefields': 
                system, topology, _ = openff_utils.get_openmm_system(mapped_smiles=mapped_smiles, openff_forcefield=forcefield)
            else:
                raise ValueError(f"forcefield_type must be either openmm, openff or openmmforcefields but is {ff_type}")

            if charge_noise > 0.:
                charges = openmm_utils.get_partial_charges(system)
                noise = np.random.normal(0, charge_noise, len(charges))
                # subtract the mean of the noise:
                noise -= np.mean(noise)
                charges += noise
                system = openmm_utils.set_partial_charges(system, charges)
                ff_name += f"_noise{charge_noise}"

            new_moldata = MolData.from_openmm_system(openmm_system=system, openmm_topology=topology, xyz=old_moldata.xyz, gradient=old_moldata.gradient, energy=old_moldata.energy, mol_id=old_moldata.mol_id, pdb=old_moldata.pdb, sequence=old_moldata.sequence, allow_nan_params=True, charge_model=charge_model, ff_name=ff_name, smiles=old_moldata.smiles, mapped_smiles=old_moldata.mapped_smiles)

            openmm_gradients = new_moldata.ff_gradient[ff_name]["total"]
            gradient = new_moldata.gradient

            # calculate the crmse:
            crmse = np.sqrt(np.mean((openmm_gradients-gradient)**2))

            if crmse > crmse_limit:
                print(f"Warning: crmse between {forcefield} and QM is {round(crmse, 3)} for {path.stem}, std of gradients is {round(np.std(gradient), 1)}")

            new_moldata.save(Path(outpath)/(path.stem+'.npz'))
        else:
            new_moldata = MolData.load(Path(outpath)/(path.stem+'.npz'))


        for contrib in CONTRIBS:
            all_gradients_new[contrib].append(new_moldata.ff_gradient[ff_name][contrib].flatten())
            all_gradients_old[contrib].append(old_moldata.ff_gradient[old_ff_name][contrib].flatten())

    # plot the different contributions:
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for i, contrib in enumerate(CONTRIBS):
        this_ax = ax[i // 3, i % 3]
        this_ax.set_title(contrib)
        grads_old = np.concatenate(all_gradients_old[contrib])
        grads_new = np.concatenate(all_gradients_new[contrib])

        scatter_plot(this_ax, grads_old, grads_new, cluster=True, logscale=True, show_rmsd=True)
        this_ax.set_xlabel(old_ff_name)
        this_ax.set_ylabel(ff_name)

    plt.tight_layout()

    plt.savefig(f"{outpath.stem}.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dspath", type=str, help="Path to the dataset")
    parser.add_argument("outpath", type=str, help="Path to the output directory")
    parser.add_argument("forcefield", type=str, help="Name of the force field")
    parser.add_argument("--ff_type", type=str, default="openmm", help="Type of the force field")
    parser.add_argument("--charge_model", type=str, default="", help="Charge model")
    parser.add_argument("--crmse_limit", type=float, default=15., help="Maximum crmse")
    parser.add_argument("--old_ff_name", type=str, default="amber99sbildn", help="Name of the old force field")
    parser.add_argument("--charge_noise", type=float, default=0., help="Add noise to the charges")
    parser.add_argument("--plot_only", action="store_true", help="Only plot the results")

    args = parser.parse_args()

    if not "\\" in args.dspath:
        args.dspath = get_data_path()/"datasets"/args.dspath

    if not "\\" in args.outpath:
        args.outpath = get_data_path()/"datasets"/args.outpath

    Path(args.outpath).mkdir(exist_ok=True, parents=True)

    reparametrize_dataset(Path(args.dspath), Path(args.outpath), args.forcefield, args.ff_type, args.charge_model, args.crmse_limit, args.old_ff_name, args.charge_noise, args.plot_only)