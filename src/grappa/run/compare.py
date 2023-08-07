
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
            if len(k.split('/')) > 2:
                raise ValueError(f"Expected dataset path to be of the form 'dataset_path/ff_info', but got '{k}'")
            if len(k.split('/')) == 1:
                continue
            ds_type = k.split('/')[0]
            
            if ds_type not in data[model_name]:
                data[model_name][ds_type] = {"energy_rmse": [], "grad_rmse": []}
            
            # Append RMSE data points to the lists
            data[model_name][ds_type]["energy_rmse"].append(data_dict[k]["energy_rmse"])
            data[model_name][ds_type]["grad_rmse"].append(data_dict[k]["grad_rmse"])


    for model_name in data.keys():
        for ds_type in data[model_name].keys():
            for metric_type in ["energy_rmse", "grad_rmse"]:
                value = copy.deepcopy(np.array(data[model_name][ds_type][metric_type]))
                data[model_name][ds_type][metric_type] = value.mean()
                data[model_name][ds_type][f"{metric_type}_std"] = value.std()

    # sort the dictionary by the energy_rmse of the first dataset:
    sorted_keys = sorted(list(data.keys()), key=lambda x: data[x][list(data[x].keys())[0]]["energy_rmse"])
    data_old = copy.deepcopy(data)
    data = {}
    for k in sorted_keys:
        data[k] = data_old[k]

    # Constants and initial data preparation
    n_models = len(list(data.keys()))
    data_types = set([d for model in data.keys() for d in data[model].keys()])
    total_width = 0.6  # adjust as needed
    single_bar_width = total_width / len(data_types)

    # Prepare the palette, ensuring the colors remain consistent across both plots
    color_palette = plt.get_cmap('tab10')
    color_list = [color_palette(i) for i in range(len(data_types))]



    # Function to create plots
    def create_plot(metric_type, filename):
        fig, ax = plt.subplots(figsize=figsize)
        r = np.arange(n_models)  # the label locations

        for model_idx, model in enumerate(data.keys()):
            for dtype_idx, dtype in enumerate(sorted(data_types)):
                value = data[model][dtype][metric_type]
                error = data[model][dtype][f"{metric_type}_std"]
                r_new = r[model_idx] + dtype_idx * single_bar_width
                ax.bar(r_new, value, color=color_list[dtype_idx], width=single_bar_width, label=f"{dtype}" if model_idx == 0 else "", yerr=error, capsize=single_bar_width*30/2)

        ax.set_ylabel('RMSE', fontsize=fontsize)
        ax.set_title(f'RMSE {metric_type.capitalize()} for Different Models and Testsets in kcal/mol (/Å)', fontsize=fontsize)
        ax.set_xticks([rp + (single_bar_width * len(data_types)) / 2 for rp in r])
        ax.set_xticklabels(list(data.keys()), rotation=45, ha="right", fontsize=fontsize)
        ax.legend(loc='best', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    plotdir = Path(plotdir)
    # Creating plots for energies and forces
    create_plot("energy_rmse", plotdir/"energy_comparision.png")
    create_plot("grad_rmse", plotdir/"forces_comparision.png")


def compare_versions(parent_dir:Union[Path,str]=Path.cwd()/"versions", fontsize:int=12, figsize:Tuple[int,int]=(10,5)):
    """
    Create a bar plot comparing model performance of models in subfolders of parent_dir.

    For each subfolder of the parent dir, check if it contains a file called 'eval_data.json'. Use the name of the subfolder as the model name. Then create a bar plot comparing the models in the parent of the parent dir.
    """
    parent_dir = Path(parent_dir)
    model_names = []
    eval_data = []
    for model_dir in parent_dir.iterdir():
        if model_dir/"eval_data.json" in model_dir.iterdir():
            logtxt = open(model_dir/"log.txt", "r").readlines()
            if not "ref_energy_rmse" in logtxt[-1]:
                print(f"Skipping {model_dir.name} because it is not finished.")
                continue

            # strip the first ..._ from the name:
            model_names.append(model_dir.name.split("_", 1)[1])
            eval_data.append(json.load(open(model_dir/"eval_data.json", "r"))["eval_data"])

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
    args = parser.parse_args()

    compare_versions(args.parent_dir, fontsize=args.fontsize, figsize=args.figsize)