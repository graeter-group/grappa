'''
This script is used to split the datasets used in grappa-1.0 as done in the experiments reported in the paper for reproduction.
Grappa uses a molecular identifier (smiles string/protein sequence) to differentiate between unique molecules that might appear several times across different datasets. Then the set of molecular ids is split and according to them, the dataset.
For easier reproduction and benchmarking of competing models, we save the splitted datasets separately in .npz format and make them available.
'''

#%%
from pathlib import Path
import json
import numpy as np
from grappa.data import Dataset, MolData

espaloma_splitpath = Path('get_espaloma_split/espaloma_split.json')

grappa_splitpath = Path('/hits/fast/mbm/seutelf/grappa/experiments/train-grappa-1.0/wandb/run-20240209_112055-atbogjqt/files/split.json')

#%%
espaloma_split = json.load(open(espaloma_splitpath))

grappa_split = json.load(open(grappa_splitpath))

# first of all, ensure consistency between the splits: the espaloma dataset is a subset of the grappa dataset, also its mol_ids should:

assert set(espaloma_split['train']) <= set(grappa_split['train'])

assert set(espaloma_split['val']) <= set(grappa_split['val'])

assert set(espaloma_split['test']) <= set(grappa_split['test'])
# %%
# now ensure consistency of the splitted molecule names (no overlaps):

assert set(grappa_split['train']) & set(grappa_split['val']) == set()

assert set(grappa_split['train']) & set(grappa_split['test']) == set()

assert set(grappa_split['val']) & set(grappa_split['test']) == set()
# %%
import yaml

# load the datasets:
configpath = grappa_splitpath.parent / 'grappa_config.yaml'
with open(configpath) as f:
    data_config = yaml.safe_load(f)["data_config"]

# load the datasets as lists of MolData objects, then split them according to the grappa split:

dspath = Path(__file__).parent.parent/'data/grappa_datasets'

splitted_ds_path = Path(__file__).parent.parent/'data/splitted_datasets'

#%%

from grappa import constants

for ds in data_config["datasets"]:
    (splitted_ds_path/'train'/ds).mkdir(parents=True, exist_ok=True)
    (splitted_ds_path/'val'/ds).mkdir(parents=True, exist_ok=True)
    (splitted_ds_path/'test'/ds).mkdir(parents=True, exist_ok=True)
    print()
    for i, entry in enumerate((dspath/ds).glob('*.npz')):
        print(f"processing {i}   : {entry.stem}", end='\r')
        md = MolData.load(entry)

        # add charge model feature by hand since it was only added to the dgl datasets so far:
        charge_model = 'classical' if ('amber99' in ds or ds == 'dipeptide_rad') else 'am1BCC'
        if not 'charge_model' in md.molecule.additional_features:
            assert charge_model in constants.CHARGE_MODELS
            md.molecule.additional_features['charge_model'] = np.tile(np.array([cm == charge_model for cm in constants.CHARGE_MODELS], dtype=np.float32), (len(self.atoms),1))
        
        dstype = None
        if md.mol_id in grappa_split['train']:
            dstype = 'train'
        elif md.mol_id in grappa_split['val']:
            dstype = 'val'
        elif md.mol_id in grappa_split['test']:
            dstype = 'test'
        else:
            raise ValueError(f"molecule {md.mol_id} not found in any split")

        molpath = splitted_ds_path/dstype/f'{ds}/{entry.stem}.npz'

        md.save(molpath)
# %%

# save the pure train/val/test sets as well:

for dstype in ['train', 'val', 'test']:
    for ds in data_config[f"pure_{dstype}_datasets"]:
        print(ds)
        (splitted_ds_path/dstype/ds).mkdir(parents=True, exist_ok=True)
        for i, entry in enumerate((dspath/ds).glob('*.npz')):
            print(f"processing {i}   : {entry.stem}", end='\r')
            md = MolData.load(entry)

            # add charge model feature by hand since it was only added to the dgl datasets so far:
            charge_model = 'classical' if ('amber99' in ds or ds == 'dipeptide_rad') else 'am1BCC'
            if not 'charge_model' in md.molecule.additional_features:
                assert charge_model in constants.CHARGE_MODELS
                md.molecule.additional_features['charge_model'] = np.tile(np.array([cm == charge_model for cm in constants.CHARGE_MODELS], dtype=np.float32), (len(self.atoms),1))
                
            molpath = splitted_ds_path/dstype/f'{ds}/{entry.stem}.npz'
            md.save(molpath)
# %%
