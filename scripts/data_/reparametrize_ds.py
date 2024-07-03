#%%
from grappa.data import MolData
from pathlib import Path
from tqdm import tqdm
from grappa.utils import openmm_utils
import numpy as np
from grappa.utils.data_utils import get_moldata_path
import matplotlib.pyplot as plt
import numpy as np
from grappa.utils.plotting import scatter_plot
from openmm.app import ForceField

this_dir = Path(__file__).parent

def reparametrize_dataset(dspath:Path, outpath:Path, forcefield:ForceField, plot_only:bool=False, ff_name:str='new', old_ff_name:str='old', crmse_limit:float=15., n_max:int=None):
    """
    Reparametrize the dataset dspath/*.npz using the force field force_field.
    """
    assert outpath != dspath
    CONTRIBS = ['bond', 'angle', 'proper', 'improper', 'nonbonded', 'total']

    all_gradients_new = {contrib: [] for contrib in CONTRIBS}
    all_gradients_old = {contrib: [] for contrib in CONTRIBS}

    assert dspath.is_dir()
    mol_data_paths = list(dspath.glob("*.npz"))
    assert len(mol_data_paths) > 0

    for path in tqdm(list(mol_data_paths)[:n_max], desc="Reparametrizing"):

        old_moldata = MolData.load(path)

        if not plot_only:
            pdbstring = old_moldata.pdb
            mapped_smiles = old_moldata.mapped_smiles

            topology = openmm_utils.topology_from_pdb(pdbstring)
            system = forcefield.createSystem(topology)

            new_moldata = MolData.from_openmm_system(openmm_system=system, openmm_topology=topology, xyz=old_moldata.xyz, gradient=old_moldata.gradient, energy=old_moldata.energy, mol_id=old_moldata.mol_id, pdb=old_moldata.pdb, sequence=old_moldata.sequence, allow_nan_params=True, ff_name=ff_name, smiles=old_moldata.smiles, mapped_smiles=old_moldata.mapped_smiles)

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

    plt.savefig(this_dir/f"{outpath.stem}.png")
    plt.close()

# %%
new_ff = ForceField('amber99sbildn.xml')

orig_ds = get_moldata_path('dipeptides-300K-amber99')

new_ds = orig_ds.parent/'dipeptides-300K-dummy'

reparametrize_dataset(orig_ds, new_ds, new_ff, ff_name='new_ff', old_ff_name='amber99sbildn', n_max=10)
# %%
ff = ForceField()
# %%
