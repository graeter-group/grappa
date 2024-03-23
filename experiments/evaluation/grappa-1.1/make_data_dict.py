#%%
from grappa.data import Dataset, GraphDataLoader
import torch
from grappa.training.evaluation import Evaluator
from pathlib import Path
import yaml
import json
import wandb
from grappa.training.resume_trainrun import get_dir_from_id
from grappa.models.deploy import model_from_config
from grappa.utils.train_utils import remove_module_prefix
from grappa.models.energy import Energy
import copy
import argparse

# Set up argument parser
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('run_id', required=True, help='The run id from wandb.')
# parser.add_argument('--project', default='benchmark-grappa-1.0', help='The project name')
# parser.add_argument('--device', default='cuda', help='The device to use')

# # Parse arguments
# args = parser.parse_args()

# PROJECT = args.project
# RUN_ID = args.run_id
# DEVICE = args.device

RUN = 'leif-seute/grappa-1.1/4te386b4'
RUN_ID = RUN.split('/')[-1]
PROJECT = RUN.split('/')[-2]
DEVICE = 'cuda'

PROJECT_DIR = Path(__file__).parent.parent.parent/'train-grappa-1.1'

WANDPATH = PROJECT_DIR/'wandb'

MODELNAME = 'best-model.ckpt'

WITH_TRAIN = False

FORCES_PER_BATCH = 2e3
BATCH_SIZE = None # if None, it will be calculated from FORCES_PER_BATCH

N_BOOTSTRAP = 1000

CLASSICAL_FF = ['amber14', 'gaff-2.11']


#%%

api = wandb.Api()
runs = api.runs(PROJECT)

rundir = get_dir_from_id(run_id=RUN_ID, wandb_folder=WANDPATH)
modelpath = str(Path(rundir)/'files'/'checkpoints'/MODELNAME)

configpath = str(Path(rundir)/'files'/'grappa_config.yaml')
#%%

# load a model without weights:
model = model_from_config(yaml.safe_load(open(configpath))["model_config"]).to(DEVICE).eval()

# add the energy layer:
model = torch.nn.Sequential(model, Energy())

state_dict = torch.load(modelpath, map_location=DEVICE)['state_dict']
state_dict = remove_module_prefix(state_dict)
model.load_state_dict(state_dict)

#%%

def get_ds_size(ds):
    return {'n_mols': len(ds), 'n_confs': sum([len(entry.nodes['g'].data['energy_ref'].flatten()) for entry, _ in ds])}

# load the datasets:
data_config = yaml.safe_load(open(configpath))["data_config"]
datasets = data_config["datasets"]

ds_size = {}

testsets = []
for tag in data_config["pure_test_datasets"]:
    ds = Dataset.from_tag(tag)
    ds_size[tag] = get_ds_size(ds)
    testsets.append(ds)

print(f'loaded pure test sets with {sum([len(ds) for ds in testsets])} molecules')

splitpath = Path(configpath).parent / f"split.json"
if not splitpath.exists():
    splitpath = Path(configpath).parent / f"files/split.json"
splitpath = splitpath if splitpath.exists() else data_config["splitpath"]

splitnames = json.load(open(splitpath))

partition = data_config["partition"]

trainsets = []

seed = 0 # used in all splits in this project

for tag in datasets:
    ds = Dataset.from_tag(tag)
    ds_size[tag] = get_ds_size(ds)

    split_ids = ds.calc_split_ids(partition=partition, seed=seed, existing_split=splitnames)

    if WITH_TRAIN:
        train_set, _, test_set = ds.split(split_ids['train'], split_ids['val'], split_ids['test'])
        trainsets.append(train_set)
    else:
        _, _, test_set = ds.split(split_ids['train'], split_ids['val'], split_ids['test'])

    testsets.append(test_set)


if WITH_TRAIN:
    pure_trainsets = []
    for tag in data_config["pure_train_datasets"]:
        ds = Dataset.from_tag(tag)
        ds_size[tag] = get_ds_size(ds)
        trainsets.append(ds)

# modify the keys. for some reason, it takes the ds path, not the tag.
ds_size_keys = copy.deepcopy(list(ds_size.keys()))
for k in ds_size_keys:
    # transform to str, then pick the last part of the path
    k_new = str(k).split('/')[-1]
    ds_size[k_new] = ds_size.pop(k)

print(json.dumps(ds_size, indent=4))

with open('ds_size.json', 'w') as f:
    json.dump(ds_size, f, indent=4)

#%%

print(f'loaded full test set with {sum([len(ds) for ds in testsets])} molecules')
result_dict = {'test': {}}

if WITH_TRAIN:
    print(f'loaded full test set with {sum([len(ds) for ds in trainsets])} molecules')
    result_dict['train'] = {}

datasets = [(True, ds) for ds in testsets]
if WITH_TRAIN:
    datasets += [(False, ds) for ds in trainsets]



###################################
for is_test, ds in datasets:
    print(ds[0][1])
    ds.remove_uncommon_features()
    confs, atoms = zip(*[(len(entry.nodes['g'].data['energy_ref']), entry.num_nodes('n1')) for entry, _ in ds])
    max_confs, max_atoms = max(confs), max(atoms)

    this_device = DEVICE
    if BATCH_SIZE is None:
        batch_size = FORCES_PER_BATCH/max_confs/max_atoms
        if batch_size < 1:
            batch_size = 1
            this_device = 'cpu'
        batch_size = int(batch_size)
    else:
        batch_size = int(max(1,BATCH_SIZE))

    model = model.to(this_device)

    loader = GraphDataLoader(ds, batch_size=batch_size, 
    conf_strategy="all", drop_last=False)

    evaluator = Evaluator()

    energy_names = list(ds[0][0].nodes['g'].data.keys())

    ff_evaluators = {}

    for ff_name in CLASSICAL_FF:
        keys_found = [k for k in energy_names if ff_name in k]
        if len(keys_found) > 1:
            raise RuntimeError(f'Found more than one key that fits ff: {keys_found}')
        elif len(keys_found) == 1:
            ff_name_in_graph = keys_found[0].replace('energy', '')
            ff_evaluators[ff_name] = Evaluator(suffix=ff_name_in_graph, suffix_ref='_qm')

    for i, (g, dsnames) in enumerate(loader):
        with torch.no_grad():
            g = g.to(this_device)
            g = model(g)
            g = g.cpu()
            print(f'batch {i+1}/{len(loader)}', end='\r')
            evaluator.step(g, dsnames)
            for k in ff_evaluators.keys():
                ff_evaluators[k].step(g, dsnames)

    print()

    d = evaluator.pool(n_bootstrap=N_BOOTSTRAP)
    d.keys()
    # d has the form {dsname:...} but since we separate all datasets, it has only one dsname:
    assert len(list(d.keys())) == 1
    dsname = list(d.keys())[0]
    d = d[dsname]

    ds_results = {
        k: d[k] for k in ['n_mols', 'n_confs', 'std_energies', 'std_gradients']
    }

    METRIC_KEYS = ['rmse_energies', 'rmse_gradients', 'crmse_gradients', 'mae_energies', 'mae_gradients']

    grappa_metrics = {
        k: d[k] for k in METRIC_KEYS
    }

    ff_metrics = {}
    for ff_name in ff_evaluators.keys():
        d = ff_evaluators[ff_name].pool(n_bootstrap=N_BOOTSTRAP)
        d = d[dsname]
        ff_metrics[ff_name] = {
            k: d[k] for k in METRIC_KEYS
        }

    # write these ff-specific results in the total dictionary:
    ds_results['grappa'] = grappa_metrics
    for k,v in ff_metrics.items():
        ds_results[k] = v

    if is_test:
        result_dict['test'][dsname]= ds_results
    else:
        result_dict['train'][dsname]= ds_results

    print(dsname, ' --- Test:' if is_test else ' --- Train:')

    print(json.dumps(ds_results, indent=4))
    print('--------------------------------------')
################################

with open('results.json', 'w') as f:
    json.dump(result_dict, f, indent=4)
# %%
    
# reformat:
data_dict = json.load(open('results.json'))
modified_dict = {}

for set_type in data_dict.keys():
    modified_dict[set_type] = {}

    for dataset in data_dict[set_type].keys():
        grappa_data = data_dict[set_type][dataset]['grappa']

        other_data = {ff: data_dict[set_type][dataset][ff] for ff in data_dict[set_type][dataset].keys() if 'grappa' not in ff}

        if '_amber99sbildn' in dataset or 'rad' in dataset:
            dataset = dataset.replace('_amber99sbildn', '')
            other_data['Grappa-ff99SB'] = grappa_data
        else:
            other_data['Grappa-AM1-BCC'] = grappa_data

        if dataset not in modified_dict[set_type]:
            modified_dict[set_type][dataset] = other_data
        else:
            modified_dict[set_type][dataset].update(other_data)
# %%
with open('results.json', 'w') as f:
    json.dump(modified_dict, f, indent=4)
# %%
