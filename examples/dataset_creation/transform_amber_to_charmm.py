#%%
"""
In this scipt, we will show how one can use grappas MolData class to obtain datasets with alternative classical force field data.
"""

#%%
from grappa.data import MolData
from grappa.utils import get_data_path
from grappa.utils.openmm_utils import get_openmm_forcefield, get_pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from grappa.utils.plotting import scatter_plot

n_data = 10

ds_path = get_data_path() / "datasets" / "dipeptides-300K-amber99"
mols = [MolData.load(path) for path in tqdm(list(ds_path.glob("*.npz"))[:n_data])]
# %%
# first, check that the stored energies are the ones from amber99:
total_gradients_stored = [mol.ff_gradient['amber99sbildn']['total'] for mol in mols]
# %%
new_mols = []
for mol in tqdm(mols):
    pdb_string = mol.pdb
    pdb = get_pdb(pdb_string)
    ff = get_openmm_forcefield("amber99sbildn")
    system = ff.createSystem(pdb.topology)
    new_mols.append(MolData.from_openmm_system(openmm_system=system, openmm_topology=pdb.topology, pdb=pdb_string, xyz=mol.xyz, energy=mol.energy, gradient=mol.gradient, ff_name="amber99sbildn", mol_id=mol.mol_id, charge_model="amber99"))
# %%
# check whether the gradients are the same
total_gradients_new = [mol.ff_gradient['amber99sbildn']['total'] for mol in new_mols]

# %%
for i in range(n_data):
    assert np.allclose(total_gradients_stored[i], total_gradients_new[i])
# %%
# now that we have validated this, we can construct a dataset with the new force field:
charmm_mols = []
for mol in tqdm(new_mols):
    pdb_string = mol.pdb
    pdb = get_pdb(pdb_string)
    ff = get_openmm_forcefield("charmm36")
    system = ff.createSystem(pdb.topology)
    charmm_mols.append(MolData.from_openmm_system(openmm_system=system, openmm_topology=pdb.topology, pdb=pdb_string, xyz=mol.xyz, energy=mol.energy, gradient=mol.gradient, ff_name="charmm36", mol_id=mol.mol_id, charge_model="charmm"))
# %%
# we can plot the different contributions:
contribs = ['bond', 'angle', 'proper', 'improper', 'nonbonded', 'total']
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for i, contrib in enumerate(contribs):
    this_ax = ax[i // 3, i % 3]
    this_ax.set_title(contrib)
    grads_amber  = np.concatenate([mol.ff_gradient['amber99sbildn'][contrib].flatten() for mol in new_mols])
    grads_charmm = np.concatenate([mol.ff_gradient['charmm36'][contrib].flatten() for mol in charmm_mols])

    scatter_plot(this_ax, grads_amber, grads_charmm)
    this_ax.set_xlabel("amber99")
    this_ax.set_ylabel("charmm36")

plt.show()
# %%
