#%%
MAX_FORCE_COMPONENT = 500 # (kcal/mol/Angstrom)

from grappa.data import MolData
from grappa.utils import get_data_path
import numpy as np
from tqdm import tqdm
import shutil

dspath_old = get_data_path() / 'datasets' / 'spice-pubchem'
dspath_new = get_data_path() / 'datasets' / 'spice-pubchem-filtered'

if dspath_new.exists():
    shutil.rmtree(dspath_new)
potential_dgl_path = dspath_new.parent.parent/'dgl_datasets'/dspath_new.name
if potential_dgl_path.exists():
    shutil.rmtree(potential_dgl_path)

#%%

# iterate over child npz files:
for npz in tqdm(list(dspath_old.glob('*.npz')), desc='Filtering states'):
    data = MolData.load(npz)
    qm_forces = data.gradient
    max_force = np.max(np.abs(qm_forces))
    if max_force > MAX_FORCE_COMPONENT:
        print(f"Filtering molecule {data.mol_id}")
        continue

        # filter out states with high forces (assume that gradients has shape (n_states, n_atoms, 3))
        delete_states = np.max(np.abs(qm_forces), axis=(1, 2)) > MAX_FORCE_COMPONENT
        # transform to indices
        delete_states = np.where(delete_states)[0]
        
        print(f"Filtering {len(delete_states)} of {len(qm_forces)} states for {npz.name} with max force {max_force}")
        if len(delete_states) == len(qm_forces):
            print(f"All states filtered out for {npz.name}")
            continue

        data.delete_states(delete_states)

    # save filtered data
    data.save(dspath_new / npz.name)
# %%