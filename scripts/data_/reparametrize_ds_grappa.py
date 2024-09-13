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
from typing import List, Tuple

this_dir = Path(__file__).parent

def reparametrize_dataset(dspath:Path, outpath:Path, forcefield:ForceField, plot_only:bool=False, ff_name:str='new', old_ff_name:str='old', crmse_limit:float=15., n_max:int=None, skip_if_exists:bool=False, other_ffs:List[Tuple[ForceField,str]]=[]):
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

        if (not plot_only) or (skip_if_exists and (Path(outpath)/(path.stem+'.npz')).exists()):
            pdbstring = old_moldata.pdb

            topology = openmm_utils.topology_from_pdb(pdbstring)
            system = forcefield.createSystem(topology)

            new_moldata = MolData.from_openmm_system(openmm_system=system, openmm_topology=topology, xyz=old_moldata.xyz, gradient=old_moldata.gradient, energy=old_moldata.energy, mol_id=old_moldata.mol_id, pdb=old_moldata.pdb, sequence=old_moldata.sequence, allow_nan_params=True, ff_name=ff_name, smiles=old_moldata.smiles, mapped_smiles=old_moldata.mapped_smiles)

            new_moldata.ff_gradient[old_ff_name] = old_moldata.ff_gradient[old_ff_name]
            new_moldata.ff_energy[old_ff_name] = old_moldata.ff_energy[old_ff_name]

            for other_ff, other_ff_name in other_ffs:
                # calc contributions for other forcefield:
                other_system = other_ff.createSystem(topology)
                other_moldata = MolData.from_openmm_system(openmm_system=other_system, openmm_topology=topology, xyz=old_moldata.xyz, gradient=old_moldata.gradient, energy=old_moldata.energy, mol_id=old_moldata.mol_id, pdb=old_moldata.pdb, sequence=old_moldata.sequence, allow_nan_params=True, ff_name=other_ff_name, smiles=old_moldata.smiles, mapped_smiles=old_moldata.mapped_smiles)

                # write to new_moldata:
                new_moldata.ff_gradient[other_ff_name] = other_moldata.ff_gradient[other_ff_name]
                new_moldata.ff_energy[other_ff_name] = other_moldata.ff_energy[other_ff_name]
                

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

    title = "Gradient Contributions [kcal/mol/Å]"
    fig.suptitle(title)

    plt.tight_layout()

    plt.savefig(this_dir/f"{outpath.stem}.png")
    plt.close()

# %%

# first, create a dataset with grappa energies and forces, save it and compare it with amber99sbildn:

from grappa import as_openmm

grappa_full = as_openmm('grappa-1.3')

orig_ds = get_moldata_path('dipeptides-300K-amber99')


new_ds_name = 'dipeptides-300K-grappa-tabulated'

new_ds = orig_ds.parent/new_ds_name

# delete new ds:
import shutil
from grappa.utils.data_utils import get_data_path
shutil.rmtree(new_ds, ignore_errors=True)
dgl_ds_path = get_data_path() / 'dgl_datasets' / new_ds_name
shutil.rmtree(dgl_ds_path, ignore_errors=True)

grappa_tabulated = ForceField('/hits/fast/mbm/hartmaec/workdir/FF99SBILDNPX_OpenMM/grappa_1-3-amber99_unique.xml')

#%%
reparametrize_dataset(orig_ds, new_ds, grappa_full, ff_name='grappa-1.3', old_ff_name='amber99sbildn', n_max=None, other_ffs=[(grappa_tabulated, 'grappa-1.3_tabulated')])
# %%
# # now, parametrize the system with another forcefield and compare to grappa:

# new_ff = ForceField('amber99sbildn.xml')

# new_ds_path = orig_ds.parent/'dipeptides-300K-new_ff'

# reparametrize_dataset(grappa_full_ds, new_ds_path, new_ff, ff_name='new_ff', old_ff_name='Grappa', n_max=10)
# # %%
