#%%
from grappa.data import Dataset, GraphDataLoader
import torch
from grappa.training.evaluation import Evaluator
from pathlib import Path
import yaml
import json
import wandb
from grappa.utils.train_utils import remove_module_prefix
from grappa.models.energy import Energy
from grappa.utils.model_loading_utils import model_dict_from_tag, url_from_tag
from grappa.training.export_model import get_model_dict
import copy
from typing import Tuple, Dict
import argparse


def grappa_eval():

    parser = argparse.ArgumentParser(description='Test a model on the datasets stored in its config file. Creates dictionaries for results of grappa and classical force fields and a dictionary for the total dataset sizes. If called with a modeltag, these dictionaries are added to (or overwrite the corresponding entries of) the modelname.pth file. Otherwise, they are stored as results.json and ds_size.json in the current directory. If executed from within a wandb run directory, the model is expected to be in the files/checkpoints/best-model.ckpt file.')
    parser.add_argument('--modeltag', '-t', type=str, help='Name of an exported grappa model. Either this or checkpoint_path or id has to be specified.', default=None)
    parser.add_argument('--checkpoint_path', '-cp', type=str, help='Absolute path to the lightning checkpoint that should be exported. Either this or modelname or id has to be specified.', default=None)
    parser.add_argument('--id', '-i', type=str, help='The wandb id of the run that should be exported. Searches for the oldest best_model.ckpt file that belongs to that run. If you use this argument, the function has to be executed from the dir that contains the wandb folder. Either this or modelname or checkpoint_path has to be specified.', default=None)
    parser.add_argument('--with_train', '-wt', action='store_true', help='If True, the training datasets are also tested.')
    parser.add_argument('--with_val', '-wv', action='store_true', help='If True, the val datasets are also tested.')
    parser.add_argument('--n_bootstrap', '-nb', type=int, default=1000, help='The number of bootstrap samples.')
    parser.add_argument('--classical_ff', '-cff', nargs='+', default=[], help='The classical force fields that should be evaluated.')
    parser.add_argument('--forces_per_batch', '-fpb', type=float, default=2e3, help='The number of forces per batch. Determines the batch size.')
    parser.add_argument('--batch_size', '-bs', type=int, default=None, help='The batch size. If None, it is calculated from forces_per_batch.')
    parser.add_argument('--device', '-d', default=None, help='The device to use. Default is cuda if available, else cpu.')

    args = parser.parse_args()

    return main_(**vars(args))


def main_(modeltag:str=None, checkpoint_path:str=None, with_train:bool=False, with_val:bool=False, n_bootstrap:int=1000, classical_ff:list=[], forces_per_batch:float=2e3, batch_size:int=None, device:str=None):

    MODELPATH = Path(__file__).parent.parent.parent.parent/'models'

    if sum([modeltag is not None, checkpoint_path is not None, id is not None]) > 1:
        raise ValueError("Either id or checkpoint_path or modeltag or none has to be specified, not all or two.")

    if not checkpoint_path is None:
        checkpoint_path = Path(checkpoint_path)
        model_dict = get_model_dict(checkpoint_path)
    elif not modeltag is None:
        model_dict = model_dict_from_tag(modeltag)
    else:
        checkpoint_path = Path.cwd() / 'files' / 'checkpoints' / 'best-model.ckpt'
        model_dict = get_model_dict(checkpoint_path)

    assert not model_dict is None, 'No model found'
    assert not model_dict['split_names'] is None, 'No split names found in the model config'

    result_dict, ds_size = eval_model(state_dict=model_dict['state_dict'], config=model_dict['config'], split_ids=model_dict['split_names'], with_train=with_train, with_val=with_val, forces_per_batch=forces_per_batch, batch_size=batch_size, n_bootstrap=n_bootstrap, classical_ff=classical_ff, device=device)

    if modeltag is not None:
        filename = str(url_from_tag(modeltag)).split('/')[-1]
        file_content = torch.load(str(MODELPATH/filename))
        file_content['results'] = result_dict
        file_content['ds_size'] = ds_size
        torch.save(file_content, str(MODELPATH/filename))
    else:
        with open('results.json', 'w') as f:
            json.dump(result_dict, f, indent=4)
        with open('ds_size.json', 'w') as f:
            json.dump(ds_size, f, indent=4)



def eval_model(state_dict: dict, config: dict, split_ids:dict, with_train:bool=False, with_val:bool=False, forces_per_batch:float=2e3, batch_size:int=None, n_bootstrap:int=1000, classical_ff:list=['amber14', 'gaff-2.11'], device:str=None)->Tuple[dict, dict]:

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # prepare the model:
    ############################################
    # model = model_from_config(config['model_config']) #NOTE
    # load the state dict:
    model.load_state_dict(state_dict)
    # add the energy layer:
    model = torch.nn.Sequential(model, Energy()).eval().to(device)
    ############################################


    # load the datasets:
    ############################################

    # helper function to get the size of the datasets:
    def get_ds_size(ds):
        return {'n_mols': len(ds), 'n_confs': sum([len(entry.nodes['g'].data['energy_ref'].flatten()) for entry, _ in ds])}

    ds_size = {}

    data_config = config["data_config"]

    # load the pure sets:
    testsets = []
    trainsets = []
    valsets = []
    for tag in data_config["pure_test_datasets"]:
        ds = Dataset.from_tag(tag)
        assert not tag in ds_size.keys(), f'Found duplicate tag {tag}'
        ds_size[tag] = get_ds_size(ds)
        testsets.append(ds)
    print(f'loaded pure test sets with {sum([len(ds) for ds in testsets])} molecules')

    if with_train:
        for tag in data_config["pure_train_datasets"]:
            ds = Dataset.from_tag(tag)
            assert not tag in ds_size.keys(), f'Found duplicate tag {tag}'
            ds_size[tag] = get_ds_size(ds)
            trainsets.append(ds)
    print(f'loaded pure train sets with {sum([len(ds) for ds in trainsets])} molecules')

    if with_val:
        for tag in data_config["pure_val_datasets"]:
            ds = Dataset.from_tag(tag)
            assert not tag in ds_size.keys(), f'Found duplicate tag {tag}'
            ds_size[tag] = get_ds_size(ds)
            valsets.append(ds)
    print(f'loaded pure val sets with {sum([len(ds) for ds in valsets])} molecules')


    # load the splitted sets and split them according to the split_ids:
    for tag in data_config["datasets"]:
        ds = Dataset.from_tag(tag)
        ds_size[tag] = get_ds_size(ds)

        train_set, val_set, test_set = ds.split(split_ids['train'], split_ids['val'], split_ids['test'])

        if with_train:
            trainsets.append(train_set)
        if with_val:
            valsets.append(val_set)

        testsets.append(test_set)


    # modify the keys, in case the tags are full paths:
    ds_size_keys = copy.deepcopy(list(ds_size.keys()))
    for k in ds_size_keys:
        # transform to str, then pick the last part of the path
        k_new = str(k).split('/')[-1]
        ds_size[k_new] = ds_size.pop(k)

    print('Full Dataset sizes:')
    print(json.dumps(ds_size, indent=4))


    print(f'loaded full test set with {sum([len(ds) for ds in testsets])} molecules')
    result_dict = {'test': {}}

    if with_train:
        print(f'loaded full train set with {sum([len(ds) for ds in trainsets])} molecules')
        result_dict['train'] = {}

    if with_val:
        print(f'loaded full val set with {sum([len(ds) for ds in valsets])} molecules')
        result_dict['val'] = {}


    ###################################
    for dstype, these_datasets in zip(['test', 'train', 'val'], [testsets, trainsets, valsets]):
        for ds in these_datasets:
            
            # determine the device and batch size:
            confs, atoms = zip(*[(len(entry.nodes['g'].data['energy_ref']), entry.num_nodes('n1')) for entry, _ in ds])
            max_confs, max_atoms = max(confs), max(atoms)
            this_device = device
            if batch_size is None:
                batch_size = forces_per_batch/max_confs/max_atoms
                if batch_size < 1:
                    batch_size = 1
                    this_device = 'cpu'
                batch_size = int(batch_size)
            else:
                batch_size = int(max(1,batch_size))

            # we hard-code this because for some reason this fails on the gpu...
            if 'pepconf' in ds[0][1]:
                this_device = 'cpu'


            print(f'Calculating metrics for {dstype} - {ds[0][1]} - {this_device}...')

            # prepare dataset for batching:
            ds.remove_uncommon_features()


            loader = GraphDataLoader(ds, batch_size=batch_size, 
            conf_strategy="all", drop_last=False)


            # evaluate the mode. allow max. one cuda oom errors and try again on cpu with higher batch size.
            finished = False
            oom_err_count = 0

            while not finished:
                try:
                    model = model.to(this_device)

                    evaluator = Evaluator(device='cpu')

                    energy_names = list(ds[0][0].nodes['g'].data.keys())

                    # create an own evaluator for each classical force field:
                    ff_evaluators = {}

                    for ff_name in classical_ff:
                        # determine the name of the ff entry in the graph:
                        keys_found = [k for k in energy_names if ff_name in k]
                        if len(keys_found) > 1:
                            raise RuntimeError(f'Found more than one key that fits ff: {keys_found}')
                        elif len(keys_found) == 1:
                            ff_name_in_graph = keys_found[0].replace('energy', '')

                            # initialize the evaluator:
                            ff_evaluators[ff_name] = Evaluator(suffix=ff_name_in_graph, suffix_ref='_qm', device='cpu')

                        # else: the ff is not present in the dataset, so we skip it

                    # iterate over the batches and calculate the metrics:
                    for i, (g, dsnames) in enumerate(loader):
                        with torch.no_grad():
                            g = g.to(this_device)
                            g = model(g)
                            g = g.cpu()
                            print(f'batch {i+1}/{len(loader)}', end='\r')
                            evaluator.step(g, dsnames)
                            for k in ff_evaluators.keys():
                                ff_evaluators[k].step(g, dsnames)

                    del loader
                    del ds
                    torch.cuda.empty_cache()

                    print()
                    with torch.no_grad():
                        d = evaluator.pool(n_bootstrap=n_bootstrap)
                    # d has the form {dsname:...} but since we separate all datasets, it has only one dsname:
                    assert len(list(d.keys())) == 1
                    dsname = list(d.keys())[0]
                    d = d[dsname]

                    finished = True

                except torch.cuda.OutOfMemoryError:
                    if oom_err_count > 0 or this_device == 'cpu':
                        raise

                    oom_err_count = 1
                    print(f'OOM error. Try again on cpu with higher batch size.')
                    this_device = 'cpu'
                    # we assume that RAM is at least 3 times bigger than VRAM, so we can use approximately a batch size that is 3 times bigger:
                    batch_size = int(3*batch_size)
                    print(f'New batch size: {batch_size}')
                    
                    # re-initialize the loader:
                    loader = GraphDataLoader(ds, batch_size=batch_size, conf_strategy="all", drop_last=False)

                    # now it will try again with the new batch size and on cpu, the next time it will raise the error if it fails again

            ds_results = {
                k: d[k] for k in ['n_mols', 'n_confs', 'std_energies', 'std_gradients']
            }

            METRIC_KEYS = ['rmse_energies', 'rmse_gradients', 'crmse_gradients', 'mae_energies', 'mae_gradients']

            grappa_metrics = {
                k: d[k] for k in METRIC_KEYS
            }

            ff_metrics = {}
            for ff_name in ff_evaluators.keys():
                with torch.no_grad():
                    d = ff_evaluators[ff_name].pool(n_bootstrap=n_bootstrap)
                d = d[dsname]
                ff_metrics[ff_name] = {
                    k: d[k] for k in METRIC_KEYS
                }

            # write these ff-specific results in the total dictionary:
            ds_results['grappa'] = grappa_metrics
            for k,v in ff_metrics.items():
                ds_results[k] = v

            result_dict[dstype][dsname]= ds_results

            print(json.dumps({met_name: ds_results['grappa'][met_name] for met_name in ['rmse_energies', 'crmse_gradients']}, indent=4))
            print('--------------------------------------')
        ################################

    return result_dict, ds_size
