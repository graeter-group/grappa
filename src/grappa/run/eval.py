import argparse
from typing import Union, List, Tuple

def eval_client():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    
    parser.add_argument('version_path', type=str, help='Path to the version folder')
    parser.add_argument('--plot', dest='plot', action='store_true', default=False, help='Plot the parameters')
    parser.add_argument('--all_loaders', dest='all_loaders', default=False, action='store_true', help='Do not plot the results')
    parser.add_argument('--ds_base', type=str, default="/hits/fast/mbm/seutelf/data/datasets/PDBDatasets", help="if tags are used, this is the base path to the datasets (default: /hits/fast/mbm/seutelf/data/datasets/PDBDatasets)")
    parser.add_argument('-t', '--ds_tag', type=str, default=[], nargs='+', help=" (dataset must be stored as dgl graphs in files named '{ds_base}/{ds_tag}_dgl.bin'. if not None, the model is evaluated on this detaset only, dividing it too into test, train and val set, keeping molecule names separated. default: [])")
    parser.add_argument('--no-forces', dest="on_forces", action='store_false', default=True, help='Only relevant if ds_tag is not None. (default: False, evaluate on forces)')
    parser.add_argument('--test', dest="test", action='store_true', default=False, help='Reducing dataset size to 50 graphs for testing functionality. (default: False)')
    parser.add_argument('--last', dest="last_model", action='store_true', default=False, help='Evaluate the last instead of the best model. (default: False)')
    parser.add_argument('--full', dest="full_loaders", action='store_true', default=False, help='Evaluate on the full loader only, without differentiating in subsets. (default: False)')
    parser.add_argument('--ds_short', default=[], type=str, nargs='+', help="codes for a collections of datasets that are added to the ds_paths. available options: \n'eric_nat': with amber charges, energies filtered at 60kcal/mol, \n'eric_rad': with amber charges (heavy), energies filtered at 60kcal/mol, \n'spice': with energies filtered at 60kcal/mol and filtered for standard amino acids, \n'eric' both of the above (default: [])")
    parser.add_argument('--collagen', dest="collagen", action='store_true', default=False, help='Evaluate on the dataset where hyp and dop are allowed. (default: False)')
    parser.add_argument('--ref_ff', '-r', type=str, default=None, help='Reference force field for the dataset. If None and amber99sbildn or amber14-all are in the ds_tag, takes this as reference forcefield. (default: None)')


    args = parser.parse_args()

    for ds_short in args.ds_short:
        suffix = "_60"
        suffix_col = ""
        if args.collagen:
            suffix_col = "_col"
        if ds_short == "eric_nat":
            args.ds_tag += [f'AA_scan_nat/amber99sbildn{suffix_col}_amber99sbildn{suffix}', f'AA_opt_nat/amber99sbildn{suffix_col}']
        if ds_short == "eric_rad":
            args.ds_tag += [f'AA_scan_rad/heavy{suffix_col}_amber99sbildn{suffix}', f'AA_opt_rad/heavy{suffix_col}_amber99sbildn{suffix}']
        if ds_short == "spice":
            args.ds_tag += [f'spice/amber99sbildn_amber99sbildn{suffix}']
        if ds_short == "eric":
            args.ds_tag += [f'AA_scan_nat/amber99sbildn{suffix_col}_amber99sbildn{suffix}', f'AA_opt_nat/amber99sbildn{suffix_col}_amber99sbildn{suffix}', f'AA_scan_rad/heavy{suffix_col}_amber99sbildn{suffix}', f'AA_opt_rad/heavy{suffix_col}_amber99sbildn{suffix}']

    vargs = vars(args)
    vargs.pop("ds_short")
    vargs.pop("collagen")

    # call the function with all arguments passed by keyword:
    eval_once(**vargs)

from grappa.run.eval_utils import evaluate
from grappa.run import run_utils
from grappa.deploy.deploy import model_from_version
from pathlib import Path
import torch
import os
import json
import pandas as pd

def eval_once(version_path, plot=True, all_loaders=False, ds_base=None, ds_tag:List[str]=None, on_forces=True, test=False, last_model=False, full_loaders=False, ref_ff:str=None):

    config_args = run_utils.load_yaml(Path(version_path)/Path("config.yml"))
    
    device = config_args["device"]
    
    model = model_from_version(version=version_path, device=device, model_name="best_model.pt" if not last_model else "last_model.pt")

    if len(ds_tag) == 0:
        return eval_on_trainset(version_path=version_path, model=model, plot=plot, all_loaders=all_loaders, on_forces=on_forces, test=test, last_model=last_model, full_loaders=full_loaders, ref_ff=ref_ff)
    else:
        return eval_on_new_set(version_path=version_path, model=model, plot=plot, ds_base=ds_base, on_forces=on_forces, test=test, ds_tag=ds_tag, device=device, ref_ff=ref_ff)



def eval_on_new_set(version_path, model, plot=True, ds_base=None, on_forces=True, test=False, ds_tag:List[str]=[], device="cpu", ref_ff:str=None):

    ds_paths = [str(Path(ds_base)/Path(t)) + "_dgl.bin" for t in ds_tag]

    ds_, _ = run_utils.get_data(ds_paths=ds_paths, n_graphs=50 if test else None)
    loaders = list(run_utils.get_loaders(ds_))

    ds_tags_ = [t.replace("/","-") for t in ds_tag]

    if ref_ff is None:
        if all("amber99sbildn" in t for t in ds_tag):
            ref_ff = "amber99sbildn"
        elif all("amber14-all" in t for t in ds_tag):
            ref_ff = "amber14-all"

    outer_name = "eval_on_custom_set"

    print(f"\n\nevaluating model in\n\t'{version_path}'\non {ds_tags_}\n")
    
    eval_data = evaluate(model=model, device=device, loaders=loaders, loader_names=ds_tags_, plot=plot, plot_folder=os.path.join(version_path,outer_name), on_forces=on_forces, verbose=True, ref_ff=ref_ff)
    
    with open(os.path.join(version_path,outer_name,"eval_data.json"), "w") as f:
        json.dump(eval_data, f, indent=4)
    print()
    print(str(pd.DataFrame(eval_data["eval_data"])))



    
def eval_on_trainset(version_path, model, plot=True, all_loaders=False, on_forces=True, test=False, last_model=False, full_loaders=False, ref_ff:str=None):
    
    config_args = run_utils.load_yaml(Path(version_path)/Path("config.yml"))
    
    device = config_args["device"]
    seed = config_args["seed"]

    outer_name = "eval_plots"
    if last_model:
        outer_name = "eval_plots_last"
    
    ds_paths = config_args["ds_path"]
    force_factor = config_args["force_factor"]
    test_ds_tags = config_args["test_ds_tags"]

    if ref_ff is None:
        if all("amber99sbildn" in t for t in ds_paths):
            ref_ff = "amber99sbildn"
        elif all("amber14-all" in t for t in ds_paths):
            ref_ff = "amber14-all"


    datasets, datanames = run_utils.get_data(ds_paths, force_factor=force_factor, n_graphs=None if not test else 50)

    # if evaluated on the train set, split the dataset according to the molecule sequences
    with open(os.path.join(version_path,"split_names.json"), "r") as f:
        splits = json.load(f)

    ds_trs, ds_vls, ds_tes, split_names = run_utils.get_splits(datasets, datanames, seed=seed, fractions=[0.8,0.1,0.1], splits=splits)


    te_loaders, te_names = run_utils.get_all_loaders(subsets=ds_tes, ds_paths=ds_paths, ds_tags=test_ds_tags, basename="te")

    if not test_ds_tags is None:
        for i in range(len(test_ds_tags)):
            test_ds_tags[i] = test_ds_tags[i].replace("/","-")


    loaders = te_loaders
    loader_names = te_names

    assert not (all_loaders and full_loaders), "only one of all_loaders and full_loaders can be true"

    if all_loaders:
        vl_loaders, vl_names = run_utils.get_all_loaders(subsets=ds_vls, ds_paths=ds_paths, ds_tags=test_ds_tags, basename="vl")
        loaders.extend(vl_loaders)
        loader_names.extend(vl_names)

        tr_loaders, tr_names = run_utils.get_all_loaders(subsets=ds_trs, ds_paths=ds_paths, ds_tags=test_ds_tags, basename="tr")
        loaders.extend(tr_loaders)
        loader_names.extend(tr_names)
    

    if full_loaders:
        loaders = [loaders[0]]
        loader_names = [loader_names[0]]

        vl_loaders, vl_names = run_utils.get_all_loaders(subsets=ds_vls, ds_paths=ds_paths, ds_tags=test_ds_tags, basename="vl")
        loaders.append(vl_loaders[0])
        loader_names.append(vl_names[0])

        tr_loaders, tr_names = run_utils.get_all_loaders(subsets=ds_trs, ds_paths=ds_paths, ds_tags=test_ds_tags, basename="tr")
        loaders.append(tr_loaders[0])   
        loader_names.append(tr_names[0])


    for i in range(len(loader_names)):
        loader_names[i] = loader_names[i].replace("/","-")

    print(f"\n\nevaluating model in\n\t'{version_path}'\non {loader_names}\n")
    
    eval_data = evaluate(model=model, device=device, loaders=loaders, loader_names=loader_names, plot=plot, plot_folder=os.path.join(version_path,outer_name), on_forces=on_forces, verbose=True, ref_ff=ref_ff)
    
    with open(os.path.join(version_path,outer_name,"eval_data.json"), "w") as f:
        json.dump(eval_data, f, indent=4)
    print()
    print(str(pd.DataFrame(eval_data["eval_data"])))

if __name__ == "__main__":
    eval_client()