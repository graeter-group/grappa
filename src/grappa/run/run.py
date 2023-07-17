from grappa.run import run_utils
from grappa.models import get_models
from grappa.training import utilities
from grappa.training.with_pretrain import train_with_pretrain
from grappa.training.utilities import get_param_statistics
from grappa.run.eval_utils import evaluate

from ..models.deploy import get_default_model_config, model_from_config

import torch
from pathlib import Path
from typing import Union, List, Tuple
import os
import json

import math

import pandas as pd


def get_default_run_config():
    args = {
        "ds_path":None,
        "storage_path":str(Path.cwd()/Path("versions")),
        "force_factor":1.,
        "energy_factor":1.,
        "param_weight":0.1,
        "confs":None,
        "mols":None,
        "seed":0,
        "test":False,
        "pretrain_steps":500,
        "train_steps":1e5,
        "patience":5e-2,
        "plots":False,
        "ref_ff":"amber99sbildn",
        "device":None,
        "description":[""],
        "lr":1e-5,
        "warmup":False,
        "name":"",
        "test_ds_tags":None,
        "recover_optimizer":False,
        "version_name":None,
        "pretrain_name":None,
    }

    return args


def run_from_config(run_config_path:Union[Path,str]=None, model_config_path:Union[Path,str]=None, idx=None, **kwargs):
    """
    Load default parameters from a config file and overwrite them with kwargs passed by the user.
    """

    # load the default args
    run_args = get_default_run_config()
    model_args = get_default_model_config()

    # overwrite the default args with those occuring in the config file or those passed by kwargs with priority for kawrgs:
    for path in (run_config_path, model_config_path):
        if not path is None:
            # load the config file (this has higher prio than the defualt args but lower than the kwargs)
            if os.path.exists(str(path)):
                config_args = run_utils.load_yaml(path)
            else:
                raise ValueError(f"Config file {str(path)} does not exist, using the passed and default args only.")
        else:
            config_args = {}

        # all arguments that are specified:
        keys = set(config_args.keys())
        keys = keys.union(set(kwargs.keys()))

        for key in keys:
            # if provided in kwargs, overwrite it
            if key in kwargs.keys():
                config_args[key] = kwargs[key]
            # write the argument in the run_args or model_args dict
            if key in run_args.keys() or key in ["load_path", "continue_path"]:
                run_args[key] = config_args[key]
            elif key in model_args.keys():
                model_args[key] = config_args[key]
            else:
                raise ValueError(f"key {key} not recognized")


    if not "ds_path" in run_args.keys():
        raise ValueError("ds_path must be specified in the config file or as a keyword argument.")
    if run_args["ds_path"] is None:
        raise ValueError("ds_path must be specified in the config file or as a keyword argument.")

    # these are not stored in the config file, we set them to None
    if not "load_path" in run_args.keys():
        run_args["load_path"] = None
    if not "continue_path" in run_args.keys():
        run_args["continue_path"] = None


    # write a config file to the version path
    run_utils.write_run_config(run_args, idx)

    run_utils.write_model_config(model_args=model_args, run_args=run_args)

    if "description" in run_args.keys():
        run_args.pop("description")
    
    if "name" in run_args.keys():
        run_args.pop("name")


    run_once(model_config=model_args, **run_args)




# param weight is wrt to the energy mse
# the zeroth data
# NOTE: use model_args dict, loaded from a config file
def run_once(storage_path, version_name, pretrain_name, model_config=get_default_model_config(), param_weight=1, confs=None, mols=None, ds_path=[None], seed=0, test=False, pretrain_steps=2e3, train_steps=1e5, patience=1e-3, plots=False, ref_ff="amber99sbildn", device=None, test_ds_tags:List[str]=None, load_path=None, lr:float=1e-6, force_factor=1., energy_factor=1., recover_optimizer=False, continue_path=None, warmup:bool=False):


    if device is None:
        device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    if not continue_path is None:
        storage_path = str(Path(continue_path).parent)
        version_name = str(Path(continue_path).name)

    start_info = f"\nstarting run, will write to:\n\t{str(Path(storage_path)/Path(version_name))}\n"
    print(start_info)

    ds_paths = ds_path

    ###################

    ###################

    n_graphs = None
    if test:
        n_graphs = 50
    datasets, datanames = run_utils.get_data(ds_paths, n_graphs=n_graphs, force_factor=force_factor)

    # the split in datanames:
    splits = None
    assert (continue_path is None or load_path is None), "not both continue_path and load_path can be specified."

    # NOTE: this is just temporarily. handle split such that the new split is again about the fraction but the test and val set remain clean.
    # load the split from the pretrained model:
    if not continue_path is None:
        with open(os.path.join(continue_path,"split_names.json"), "r") as f:
            splits = json.load(f)

    elif not load_path is None:
        with open(os.path.join(load_path,"split_names.json"), "r") as f:
            splits = json.load(f)
    

    ds_trs, ds_vls, ds_tes, split_names = run_utils.get_splits(datasets, datanames, seed=seed, fractions=[0.8,0.1,0.1], splits=splits)
    with open(os.path.join(storage_path,version_name,"split_names.json"), "w") as f:
        json.dump(split_names, f, indent=4)


    ds_tr, ds_vl, ds_te = run_utils.flatten_splits(ds_trs, ds_vls, ds_tes)

    # make the number of molecules smaller if it is None or larger than the train set
    if not mols is None:
        run_utils.reduce_mols(ds_tr, mols, seed=seed)

    if not confs is None:
        run_utils.reduce_confs(ds_tr, confs, seed=seed)

    if not mols is None and mols > len(ds_tr):
        mols = len(ds_tr)

    tr_loader, vl_loader = run_utils.get_loaders((ds_tr, ds_vl))

    # only use the training set for statistics
    statistics = get_param_statistics(loader=tr_loader)

    # initialize the model
    model = model_from_config(config=model_config, stat_dict=statistics)

    ###################
    pretrain_epochs = math.ceil(pretrain_steps/len(ds_tr))
    epochs = int(train_steps/len(ds_tr))
    patience_ = int(patience*epochs)

    ###################

    # do the actual training (this function is a mess, will be cleaned up)
    train_with_pretrain(model, version_name, pretrain_name, tr_loader, vl_loader, storage_path=storage_path, patience=patience_, epochs=epochs, energy_factor=energy_factor, force_factor=force_factor, lr_conti=lr, lr_pre=1e-4, device=device, ref_ff=ref_ff, pretrain_epochs=pretrain_epochs, param_factor=param_weight, param_statistics=statistics, classification_epochs=3, direct_eval=False, final_eval=False, reduce_factor=0.5, load_path=load_path, recover_optimizer=recover_optimizer, continue_path=continue_path, use_warmup=warmup)

    ###################

    # do final evaluation
    model_path = os.path.join(storage_path,version_name,"best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(storage_path,version_name,"last_model.pt")
    if not os.path.exists(model_path):
        raise ValueError(f"no model found at {model_path}")
    
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    te_loaders, te_names = run_utils.get_all_loaders(subsets=ds_tes, ds_paths=ds_paths, ds_tags=test_ds_tags, basename="te")

    on_forces = False if force_factor == 0 else True

    print(f"\n\nevaluating model\n\t'{model_path}'\non tr, val and test set...\n")

    for n in te_names:
        n = n.replace("/","-")

    plot_folder = os.path.join(storage_path,version_name,"final_rmse_plots")

    eval_data = evaluate(loaders=[tr_loader, vl_loader]+te_loaders, loader_names=["tr", "val"]+te_names, model=model, device=device, plot=False, on_forces=on_forces, plot_folder=plot_folder, rmse_plots=True)
    
    with open(os.path.join(storage_path,version_name,"eval_data.json"), "w") as f:
        json.dump(eval_data, f, indent=4)
    print()

    print(str(pd.DataFrame(eval_data["eval_data"])))

    # write the data of the full loaders to the log file:
    logdata = "\n"
    full_loaders_dict = {}
    for k in eval_data["eval_data"]:
        if k in ["tr", "val", "vl", "te"]:
            full_loaders_dict[k] = eval_data["eval_data"][k]
    logdata += str(pd.DataFrame(full_loaders_dict))
    with open(os.path.join(storage_path,version_name,"log.txt"), "a") as f:
        f.write(logdata)

    if plots:
        print("plotting...")

        loader_names=["tr", "val"]+te_names

        for i in range(len(loader_names)):
            loader_names[i] = loader_names[i].replace("/","-")


        evaluate(loaders=[tr_loader, vl_loader]+te_loaders, loader_names=loader_names, model=model, device=device, plot_folder=os.path.join(storage_path,version_name,"final_eval_plots"), plot=True, metrics=False, on_forces=True, rmse_plots=False)