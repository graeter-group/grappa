import argparse
from typing import Union, List, Tuple
from grappa.PDBData.PDBDataset import PDBDataset, SplittedDataset

def eval_client():
    parser = argparse.ArgumentParser(description='Evaluate a model')
    
    parser.add_argument('version_path', type=str, help='Path to the version folder')
    parser.add_argument('--plot', dest='plot', action='store_true', default=False, help='Plot the parameters')
    parser.add_argument('--all_loaders', dest='all_loaders', default=False, action='store_true', help='eval on all loaders')
    parser.add_argument('--ds_base', type=str, default="/hits/fast/mbm/seutelf/data/datasets/PDBDatasets", help="if tags are used, this is the base path to the datasets (default: /hits/fast/mbm/seutelf/data/datasets/PDBDatasets)")
    parser.add_argument('-t', '--ds_tag', type=str, default=[], nargs='+', help=" (dataset must be stored in files named '{ds_base}/{ds_tag}'. if not None, the model is evaluated on this dataset only, dividing it too into test, train and val set, keeping molecule names separated. default: [])")
    parser.add_argument('--no-forces', dest="on_forces", action='store_false', default=True, help='Only relevant if ds_tag is not None. (default: False, evaluate on forces)')
    parser.add_argument('--test', dest="test", action='store_true', default=False, help='Reducing dataset size to 50 graphs for testing functionality. (default: False)')
    parser.add_argument('--last', dest="last_model", action='store_true', default=False, help='Evaluate the last instead of the best model. (default: False)')
    parser.add_argument('--full', dest="full_loaders", action='store_true', default=False, help='Evaluate on the full loader only, without differentiating in subsets. (default: False)')
    parser.add_argument('--ds_short', default=[], type=str, nargs='+', help="codes for a collections of datasets that are added to the ds_paths. available options: \n'eric_nat': with amber charges, energies filtered at 60kcal/mol, \n'eric_rad': with amber charges (heavy), energies filtered at 60kcal/mol, \n'spice': with energies filtered at 60kcal/mol and filtered for standard amino acids, \n'eric' both of the above (default: [])")
    parser.add_argument('--collagen', dest="collagen", action='store_true', default=False, help='Evaluate on the dataset where hyp and dop are allowed. (default: False)')
    parser.add_argument('--ref_ff', '-r', type=str, default="ref", help='Reference force field for the dataset. If None and amber99sbildn or amber14-all are in the ds_tag, takes this as reference forcefield. (default: ref)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use. (default: cuda)')


    args = parser.parse_args()

    for ds_short in args.ds_short:
        suffix = "_filtered"
        suffix_col = ""
        if args.collagen:
            suffix_col = "_col"

        if ds_short == "eric_nat":
            args.ds_tag += [f'AA_scan_nat/charge_amber99sbildn{suffix_col}_ff_amber99sbildn{suffix}', f'AA_opt_nat/charge_amber99sbildn{suffix_col}_ff_amber99sbildn{suffix}']

        if ds_short == "eric_rad":
            args.ds_tag += [f'AA_scan_rad/charge_heavy{suffix_col}_ff_amber99sbildn{suffix}', f'AA_opt_rad/charge_heavy{suffix_col}_ff_amber99sbildn{suffix}']

        if ds_short == "spice":
            args.ds_tag += [f'spice/charge_default_ff_amber99sbildn{suffix}']

        if ds_short == "spice_openff":
            args.ds_tag += [f'spice_openff/charge_default_ff_gaff-2_11{suffix}']

        if ds_short == "spice_monomers":
            args.ds_tag += [f'monomers/charge_default_ff_gaff-2_11{suffix}']

        if ds_short == "eric":
            args.ds_short.remove("eric")
            args.ds_short += ["eric_nat", "eric_rad"]

    vargs = vars(args)
    vargs.pop("ds_short")
    vargs.pop("collagen")

    # call the function with all arguments passed by keyword:
    eval_once(**vargs)

from grappa.run.eval_utils import evaluate
from grappa.run import run_utils
from grappa.models.deploy import model_from_version
from pathlib import Path
import torch
import os
import json
import pandas as pd

def eval_once(version_path, plot=True, all_loaders=False, ds_base=None, ds_tag:List[str]=None, on_forces=True, test=False, last_model=False, full_loaders=False, ref_ff:str="ref", device=None):

    if device is None:
        device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    model = model_from_version(version=version_path, device=device, model_name="best_model.pt" if not last_model else "last_model.pt")

    if len(ds_tag) == 0:
        return eval_on_trainset(version_path=version_path, model=model, plot=plot, all_loaders=all_loaders, on_forces=on_forces, test=test, last_model=last_model, full_loaders=full_loaders, ref_ff=ref_ff, device=device)
    else:
        return eval_on_new_set(version_path=version_path, model=model, plot=plot, ds_base=ds_base, on_forces=on_forces, test=test, ds_tag=ds_tag, device=device, ref_ff=ref_ff)



def eval_on_new_set(version_path, model, plot=True, ds_base=None, on_forces=True, test=False, ds_tag:List[str]=[], device="cpu", ref_ff:str=None):

    ds_paths = [str(Path(ds_base)/Path(t)) for t in ds_tag]

    n_graphs = 50 if test else None

    # load the datasets
    print(f"loading: \n{ds_paths}")
    datasets = [PDBDataset.load_npz(path, n_max=n_graphs) for path in ds_paths]

    ds_splitter = SplittedDataset.load_from_names(str(Path(version_path)/Path("split")), datasets=datasets, split=[0,0,1.])

    loaders = [ds_splitter.get_loaders(i)[2] for i in range(len(ds_tag))]

    ds_tags_ = [t.replace("/","-") for t in ds_tag]


    outer_name = "eval_on_custom_set"

    print(f"\n\nevaluating model in\n\t'{version_path}'\non {ds_tags_}\n")
    
    eval_data = evaluate(model=model, device=device, loaders=loaders, loader_names=ds_tags_, plot=plot, plot_folder=os.path.join(version_path,outer_name), on_forces=on_forces, verbose=True, ref_ff=ref_ff)
    
    with open(os.path.join(version_path,outer_name,"eval_data.json"), "w") as f:
        json.dump(eval_data, f, indent=4)
    print()
    print(str(pd.DataFrame(eval_data["eval_data"])))



    
def eval_on_trainset(version_path, model, plot=True, all_loaders=False, on_forces=True, test=False, last_model=False, full_loaders=False, ref_ff:str="ref", device="cpu"):
    
    config_args = run_utils.load_yaml(Path(version_path)/Path("run_config.yml"))
    
    seed = config_args["seed"]

    outer_name = "eval_plots"
    if last_model:
        outer_name = "eval_plots_last"
    
    ds_paths = config_args["ds_path"]
    force_factor = 0
    test_ds_tags = config_args["test_ds_tags"]

    n_graphs = 50 if test else None

    # load the datasets
    print(f"loading: \n{ds_paths}")
    datasets = [PDBDataset.load_npz(path, n_max=n_graphs) for path in ds_paths]

    
    ds_splitter = SplittedDataset.load(str(Path(version_path)/Path("split")), datasets=datasets)


    # if not confs is None:
    #     run_utils.reduce_confs(ds_tr, confs, seed=seed)

    te_loaders = [ds_splitter.get_loaders(i)[2] for i in range(len(datasets))]


    if not test_ds_tags is None:
        for i in range(len(test_ds_tags)):
            test_ds_tags[i] = test_ds_tags[i].replace("/","-")

    te_names = test_ds_tags if not test_ds_tags is None else [f"ds_{i}" for i in range(len(te_loaders))]

    assert not (all_loaders and full_loaders), "only one of all_loaders and full_loaders can be true"

    if all_loaders:
        loaders = []
        loader_names = []
        for i in range(len(datasets)):
            loaders += ds_splitter.get_loaders(i)
            loader_names += [f"{te_names[i]}_train", f"{te_names[i]}_val", f"{te_names[i]}_test"]

    elif full_loaders:
        loaders = ds_splitter.get_full_loaders()
        loader_names = ["train", "val", "test"]

    else:
        loaders = te_loaders
        loader_names = te_names

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