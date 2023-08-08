
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
import os
import copy

def compare(eval_data:List[Dict], model_names:List[str], plotdir:Union[Path,str]=Path.cwd(), fontsize:int=12, figsize:Tuple[int,int]=(10,5)):
    """
    Assume the dicts in eval_data have the form
        eval_data[dataset_path/ff_info]['energy_rmse'] = float
        eval_data[dataset_path/ff_info]['grad_rmse'] = float
    Then this function creates a bar plot with bars for each model for each dataset_path with the rmse values on the y axis.
    """

    data = {}
    for data_dict, model_name in zip(eval_data, model_names):
        if model_name not in data:
            data[model_name] = {}
        for k in data_dict.keys():
            if len(k.split('>')) > 2:
                raise ValueError(f"Expected dataset path to be of the form 'dataset_path/ff_info', but got '{k}'")
            if len(k.split('>')) == 1:
                continue
            
            set_type = "train" if "train" in k else None
            set_type = "val" if "val" in k else set_type
            set_type = "test" if "test" in k else set_type

            if set_type is None:
                continue


            ds_type = k.split('>')[0]

            
            if ds_type not in data[model_name].keys():
                empty_subdict = {"energy_rmse": [], "grad_rmse": []}
                data[model_name][ds_type] = {"train": copy.deepcopy(empty_subdict), "val": copy.deepcopy(empty_subdict), "test": copy.deepcopy(empty_subdict)}

            
            # Append RMSE data points to the lists
            data[model_name][ds_type][set_type]["energy_rmse"].append(data_dict[k]["energy_rmse"])
            data[model_name][ds_type][set_type]["grad_rmse"].append(data_dict[k]["grad_rmse"])


    for model_name in data.keys():
        for ds_type in data[model_name].keys():
            for set_type in data[model_name][ds_type].keys():
                for metric_type in ["energy_rmse", "grad_rmse"]:
                    value = copy.deepcopy(np.array(data[model_name][ds_type][set_type][metric_type]))
                    data[model_name][ds_type][set_type][metric_type] = value.mean()
                    data[model_name][ds_type][set_type][f"{metric_type}_std"] = value.std()

    # sort the dictionary by the test energy_rmse of the first dataset:
    assert not 0 in [len(list(data[k].keys())) for k in data.keys()], f"Expected all models to have data for at least one dataset."
    sorted_keys = sorted(list(data.keys()), key=lambda x: data[x][list(data[x].keys())[0]]["test"]["energy_rmse"])
    data_old = copy.deepcopy(data)
    data = {}
    for k in sorted_keys:
        data[k] = data_old[k]

    # data is now a dictionary of the form
    # data[model_name][ds_type][set_type][metric_type] = float
    # for set_type in ["train", "val", "test"]:


    # Constants and initial data preparation
    n_models = len(list(data.keys()))
    data_types = set([d for model in data.keys() for d in data[model].keys()])
    if len(data_types) == 0:
        raise ValueError(f"No data found in eval_data.")
    total_width = 0.6  # adjust as needed
    single_bar_width = total_width / len(data_types)

    # Prepare the palette, ensuring the colors remain consistent across both plots
    color_palette = plt.get_cmap('tab10')
    color_list = [color_palette(i) for i in range(len(data_types))]

    def create_plot_for_sets(metric_type, filename):
        fig, ax = plt.subplots(figsize=figsize)
        r = np.arange(n_models)  # the label locations

        for model_idx, model in enumerate(data.keys()):
            for dtype_idx, dtype in enumerate(sorted(data_types)):
                value = data[model][dtype]["test"][metric_type]
                error = data[model][dtype]["test"][f"{metric_type}_std"]
                r_new = r[model_idx] + dtype_idx * single_bar_width
                ax.bar(r_new, value, color=color_list[dtype_idx], width=single_bar_width, label=f"{dtype}" if model_idx == 0 else "", yerr=error, capsize=single_bar_width*30/2)

        ax.set_ylabel('RMSE', fontsize=fontsize)
        ax.set_title(f'Test RMSE {metric_type.capitalize()} for Different Models in kcal/mol (/Å)', fontsize=fontsize)
        ax.set_xticks([rp + (single_bar_width * len(data_types)) / 2 for rp in r])
        ax.set_xticklabels(list(data.keys()), rotation=45, ha="right", fontsize=fontsize)
        ax.legend(loc='best', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def create_combined_plot(metric_type, filename, dtype, dtype_idx):
        set_types = ["train", "test"]
        fig, ax = plt.subplots(figsize=figsize)
        r = np.arange(n_models)

        for model_idx, model in enumerate(data.keys()):
            for set_idx, set_type in enumerate(set_types):
                value = data[model][dtype][set_type][metric_type]
                error = data[model][dtype][set_type][f"{metric_type}_std"]
                r_new = r[model_idx] + dtype_idx * single_bar_width * len(set_types) + set_idx * single_bar_width
                alpha = 1 if set_type == "test" else 0.5
                ax.bar(r_new, value, color=color_list[dtype_idx], width=single_bar_width, label=f"{dtype} {set_type}" if model_idx == 0 else "", yerr=error, capsize=single_bar_width*30/2, alpha=alpha)

        ax.set_ylabel('RMSE', fontsize=fontsize)
        mtype = "Energy" if "energy" in metric_type.lower() else "Force"
        ax.set_title(f'{mtype} RMSE for Different Models in kcal/mol{("/Å" if mtype=="Force" else "")}', fontsize=fontsize)
        ax.set_xticks([rp + len(set_types) * (single_bar_width * len(data_types)) / 2 for rp in r])
        ax.set_xticklabels(list(data.keys()), rotation=45, ha="right", fontsize=fontsize)
        ax.legend(loc='best', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    plotdir = Path(plotdir)

    # Creating plots for the test set
    create_plot_for_sets("energy_rmse", plotdir/"test_energy_comparision.png")
    create_plot_for_sets("grad_rmse", plotdir/"test_forces_comparision.png")

    for dtype_idx, dtype in enumerate(sorted(data_types)):
        # Creating plots for the train, val, and test set
        create_combined_plot("energy_rmse", plotdir/f"{dtype}_combined_energy_comparision.png", dtype, dtype_idx)
        create_combined_plot("grad_rmse", plotdir/f"{dtype}_combined_forces_comparision.png", dtype, dtype_idx)


    with open(plotdir/"data.json", "w") as f:
        json.dump(data, f, indent=4)


def compare_versions(parent_dir:Union[Path,str]=Path.cwd()/"versions", fontsize:int=12, figsize:Tuple[int,int]=(10,5), vpaths:List[str]=None):
    """
    Create a bar plot comparing model performance of models in subfolders of parent_dir.

    For each subfolder of the parent dir, check if it contains a file called 'eval_data.json'. Use the name of the subfolder as the model name. Then create a bar plot comparing the models in the parent of the parent dir.
    """
    parent_dir = Path(parent_dir)
    model_names = []
    eval_data = []
    dirs = parent_dir.iterdir()
    if not vpaths is None:
        dirs = vpaths
    for model_dir in dirs:

        model_dir_ = Path(model_dir)
        
        # if grappa_eval was called, the file is here:
        if (model_dir_/"eval_plots"/"eval_data.json").exists():
            model_dir_ = model_dir_/"eval_plots"

        if model_dir_/"eval_data.json" in model_dir_.iterdir():
            logtxt = open(model_dir/"log.txt", "r").readlines()
            if not "ref_energy_rmse" in logtxt[-1] and vpaths is None:
                print(f"Skipping {model_dir.name} because it is not finished.")
                continue

            # strip the first ..._ from the name:
            model_names.append(model_dir.name.split("_", 1)[1] if len(model_dir.name.split("_", 1)) > 1 else model_dir.name)
            eval_data.append(json.load(open(model_dir_/"eval_data.json", "r"))["eval_data"])

    compare(eval_data, model_names, plotdir=parent_dir.parent, fontsize=fontsize, figsize=figsize)


def client():
    """
    Create a bar plot comparing model performance of models in subfolders of parent_dir.

    For each subfolder of the parent dir, check if it contains a file called 'eval_data.json'. Use the name of the subfolder as the model name. Then create a bar plot comparing the models in the parent of the parent dir.
    """
    parser = argparse.ArgumentParser(description="Compare the performance of different models on different datasets.")

    parser.add_argument("-d", "--parent_dir", type=str, help="The parent directory containing the models to compare.", default=Path.cwd()/"versions")
    parser.add_argument("--fontsize", type=int, default=12, help="The fontsize for the plot.")
    parser.add_argument("--figsize", type=int, nargs=2, default=(10,5), help="The figure size for the plot.")
    parser.add_argument("--vpaths", type=str, nargs="+", default=None, help="The paths to the versions to compare.")
    args = parser.parse_args()

    compare_versions(args.parent_dir, fontsize=args.fontsize, figsize=args.figsize, vpaths=args.vpaths)