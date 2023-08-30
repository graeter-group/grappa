
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union
import os
import copy
from grappa.run.run_utils import load_yaml

def make_data_list(eval_data:List[Dict], reference=False):
    """
    Modify the dict keys such that the keys are: [ds_type][set_type]['energy_rmse'/'grad_rmse']:
    """
    outlist = []
    for data_dict in eval_data:
        outdict = {}
        for k in data_dict.keys():
            # if len(k.split('>')) > 2:
            #     raise ValueError(f"Expected dataset path to be of the form 'dataset_path/ff_info', but got '{k}'")
            # if len(k.split('>')) == 1:
            #     continue

            if len(k.split('_')) < 2:
                continue

            
            set_type = "train" if "train" in k else None
            set_type = "val" if "val" in k else set_type
            set_type = "test" if "test" in k else set_type

            if set_type is None:
                continue
            # all but the last _...:
            # concat all splits but the last:
            ds_type = "_".join(k.split("_")[:-1])

            if ds_type not in outdict.keys():
                empty_subdict = {"energy_rmse": [], "grad_rmse": []}
                outdict[ds_type] = {"train": copy.deepcopy(empty_subdict), "val": copy.deepcopy(empty_subdict), "test": copy.deepcopy(empty_subdict)}

            if not reference:
                # Append RMSE data points to the lists
                outdict[ds_type][set_type]["energy_rmse"] = data_dict[k]["energy_rmse"]
                outdict[ds_type][set_type]["grad_rmse"] = data_dict[k]["grad_rmse"]

            else:
                # Append RMSE data points to the lists
                outdict[ds_type][set_type]["energy_rmse"] = data_dict[k]["ref_energy_rmse"]
                outdict[ds_type][set_type]["grad_rmse"] = data_dict[k]["ref_grad_rmse"]

        outlist.append(outdict)
    
    return outlist


def make_data_dict(eval_data:List[Dict], model_names:List[str]):
    """
    assume that eval_data is the output of make_data_list
    """
    data = {}
    for data_dict, model_name in zip(eval_data, model_names):
        if model_name not in data:
            data[model_name] = {}
        for ds_type in data_dict.keys():
            if ds_type not in data[model_name].keys():
                data[model_name][ds_type] = {}
            for set_type in data_dict[ds_type].keys():
                if set_type not in data[model_name][ds_type].keys():
                    data[model_name][ds_type][set_type] = {"energy_rmse": [], "grad_rmse": []}
            
                # Append RMSE data points to the lists
                data[model_name][ds_type][set_type]["energy_rmse"].append(data_dict[ds_type][set_type]["energy_rmse"])
                data[model_name][ds_type][set_type]["grad_rmse"].append(data_dict[ds_type][set_type]["grad_rmse"])

    return data


def compare(eval_dicts:Dict, model_names:List[str], plotdir:Union[Path,str]=Path.cwd(), fontsize:int=12, figsize:Tuple[int,int]=(10,5), best=False, folds=False):
    """
    Assume the dicts in eval_data have the form
        eval_data[dataset_path/ff_info]['energy_rmse'] = float
        eval_data[dataset_path/ff_info]['grad_rmse'] = float
    Then this function creates a bar plot with bars for each model for each dataset_path with the rmse values on the y axis.
    best flag make sonly sense i all runs are on the same dataset split seed.
    """

    fontname = "Arial"

    import matplotlib as mpl
    from matplotlib.font_manager import findSystemFonts
    available_fonts = [f.split('/')[-1] for f in findSystemFonts()]
    if any(fontname in f for f in available_fonts):
        mpl.rc('font', family=fontname)
    else:
        mpl.rc('font', family='DejaVu Sans')


    data = make_data_dict(eval_dicts, model_names)

    for model_name in data.keys():
        for ds_type in data[model_name].keys():
            for set_type in data[model_name][ds_type].keys():
                for metric_type in ["energy_rmse", "grad_rmse"]:
                    value = copy.deepcopy(np.array(data[model_name][ds_type][set_type][metric_type]))
                    data[model_name][ds_type][set_type][metric_type] = value.mean()
                    data[model_name][ds_type][set_type][f"{metric_type}_std"] = value.std()
                    data[model_name][ds_type][set_type][f"{metric_type}_best"] = value.min()

    # sort the dictionary by the test energy_rmse of the first dataset:
    assert not 0 in [len(list(data[k].keys())) for k in data.keys()], f"Expected all models to have data for at least one dataset."
    # sorted_keys = sorted(list(data.keys()), key=lambda x: data[x][list(data[x].keys())[0]]["test"]["energy_rmse"])
    # data_old = copy.deepcopy(data)
    # data = {}
    # for k in sorted_keys:
    #     data[k] = data_old[k]

    # data is now a dictionary of the form
    # data[model_name][ds_type][set_type][metric_type] = float
    # for set_type in ["train", "val", "test"]:


    # Constants and initial data preparation
    n_models = len(list(data.keys()))
    data_types = set([d for model in data.keys() for d in data[model].keys()])
    if len(data_types) == 0:
        raise ValueError(f"No data found in eval_data.")
    total_width = 0.6  # adjust as needed

    # Prepare the palette, ensuring the colors remain consistent across both plots
    color_palette = plt.get_cmap('tab10')
    color_list = [color_palette(i) for i in range(len(data_types))]

    def create_plot_for_sets(metric_type, filename):
        single_bar_width = total_width / len(data_types)
        fig, ax = plt.subplots(figsize=figsize)
        r = np.arange(n_models)  # the label locations

        mnames = []
        for model_idx, model in enumerate(data.keys()):
            mnames.append(model)
            for dtype_idx, dtype in enumerate(sorted(data_types)):
                if best:
                    if dtype in data[model].keys():
                        value = data[model][dtype]["test"][f"{metric_type}_best"]
                    else:
                        value = 0
                    error = 0
                else:
                    if dtype in data[model].keys():
                        value = data[model][dtype]["test"][metric_type]
                        error = data[model][dtype]["test"][f"{metric_type}_std"]
                    else:
                        value = 0
                        error = 0

                r_new = r[model_idx] + dtype_idx * single_bar_width
                ax.bar(r_new, value, color=color_list[dtype_idx], width=single_bar_width, label=(f"{dtype.replace('_', ' ')}" if not folds else dtype.capitalize()) if model_idx == 0 else "", yerr=error, capsize=single_bar_width*30/2)

        ax.set_ylabel('RMSE', fontsize=fontsize)
        if folds:
            ax.set_title(f'Test RMSE {metric_type.capitalize()} for Different Folds in kcal/mol (/Å)', fontsize=fontsize)
        else:
            ax.set_title(f'Test RMSE {metric_type.capitalize()} for Different Models in kcal/mol (/Å)', fontsize=fontsize)
        offset = single_bar_width * (len(data_types)-1.)/2.
        ax.set_xticks((r + offset).tolist())
        ax.set_xticklabels(mnames, rotation=45, ha="right", fontsize=fontsize)
        ax.legend(loc='best', fontsize=fontsize)
        # make a grid on the plot
        ax.grid(True, axis='y', linestyle='--', alpha=1)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    def create_combined_plot(metric_type, filename, dtype, dtype_idx):
        set_types = ["train", "test"]
        single_bar_width = total_width / len(set_types)
        fig, ax = plt.subplots(figsize=figsize)
        r = np.arange(n_models)

        mnames = []
        for model_idx, model in enumerate(data.keys()):
            mnames.append(model)
            for set_idx, set_type in enumerate(set_types):
                if best:
                    if set_type == "test":
                        value = data[model][dtype][set_type][f"{metric_type}_best"]
                    else:
                        value = 0
                    error = 0
                else:
                    if dtype in data[model].keys():
                        value = data[model][dtype][set_type][metric_type]
                        error = data[model][dtype][set_type][f"{metric_type}_std"]
                    else:
                        value = 0
                        error = 0
                r_new = r[model_idx] + set_idx * single_bar_width
                alpha = 1 if set_type == "test" else 0.5
                ax.bar(r_new, value, color=color_list[dtype_idx], width=single_bar_width, label=(f"{dtype.replace('_', ' ')} {set_type}" if not folds else set_type.capitalize()) if model_idx == 0 else "", yerr=error, capsize=single_bar_width*30/2, alpha=alpha)

        ax.set_ylabel('RMSE', fontsize=fontsize)
        mtype = "Energy" if "energy" in metric_type.lower() else "Force"
        if folds:
            ax.set_title(f'{mtype} RMSE for Different Folds in kcal/mol{("/Å" if mtype=="Force" else "")}', fontsize=fontsize)
        else:
            ax.set_title(f'{mtype} RMSE for Different Models in kcal/mol{("/Å" if mtype=="Force" else "")}', fontsize=fontsize)
        offset = (len(set_types)-1.)*single_bar_width/2.
        ax.set_xticks((r + offset).tolist())
        ax.set_xticklabels(mnames, rotation=45, ha="right", fontsize=fontsize)
        
        if folds:
            if dtype == "spice_monomers" and "energy" in metric_type.lower():
                # plot red vertical line at 1.68:
                ax.axhline(y=1.68, color="red", linestyle="--", label="Espaloma", alpha=0.5)
                ax.set_title(f'{mtype} RMSE for Different Folds of Spice Monomers in kcal/mol{("/Å" if mtype=="Force" else "")}', fontsize=fontsize)

            if dtype == "spice_qca" and "energy" in metric_type.lower():
                # plot red vertical line at
                ax.axhline(y=3.15, color="red", linestyle="--", label="Espaloma", alpha=0.5)
                ax.set_title(f'{mtype} RMSE for Different Folds of Spice Dipeptides in kcal/mol{("/Å" if mtype=="Force" else "")}', fontsize=fontsize)

            if dtype == "spice_pubchem" and "energy" in metric_type.lower():
                # plot red vertical line at
                ax.axhline(y=2.52, color="red", linestyle="--", label="Espaloma", alpha=0.5)
                ax.set_title(f'{mtype} RMSE for Different Folds of Spice Pubchem in kcal/mol{("/Å" if mtype=="Force" else "")}', fontsize=fontsize)



        ax.legend(loc='best', fontsize=fontsize)
        ax.grid(True, axis='y', linestyle='--', alpha=1)


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


def compare_versions(parent_dir:Union[Path,str]=Path.cwd()/"versions", fontsize:int=12, figsize:Tuple[int,int]=(10,5), vpaths:List[str]=None, exclude:List[str]=[], best_criterion:Tuple[str, str]=None, refname="ref", folds=False, best=False, ref=False):
    """
    Create a bar plot comparing model performance of models in subfolders of parent_dir.

    For each subfolder of the parent dir, check if it contains a file called 'eval_data.json'. Use the name of the subfolder as the model name. Then create a bar plot comparing the models in the parent of the parent dir.
    best_criterion: (metric, ds_type)
    """
    parent_dir = Path(parent_dir)
    model_names = []
    eval_data = []
    splits = []
    dirs = parent_dir.iterdir()

    existing_folds = []

    from grappa.run.run_utils import load_yaml

    if not vpaths is None:
        dirs = vpaths
    for model_dir in dirs:

        model_dir_ = Path(model_dir)

        run_config = load_yaml(model_dir/"run_config.yml")

        foldpath = run_config["ds_split_names"]

        if foldpath not in existing_folds:
            existing_folds.append(foldpath)

        
        # if grappa_eval was called, the file is here:
        if (model_dir_/"eval_plots"/"eval_data.json").exists():
            model_dir_ = model_dir_/"eval_plots"

        if model_dir_/"eval_data.json" in model_dir_.iterdir():
            logtxt = open(model_dir/"log.txt", "r").readlines()
            is_finished = "ref_energy_rmse" in logtxt[-1] or "rows" in logtxt[-1]
            if not is_finished and vpaths is None:
                print(f"Skipping {model_dir.name} because it is not finished.")
                continue

            # strip the first ..._ from the name:
            model_name = model_dir.name.split("_", 1)[1] if len(model_dir.name.split("_", 1)) > 1 else model_dir.name

            if folds:
                # the index of foldpath in the list of existing folds:
                model_name = f"{existing_folds.index(foldpath)}"

            if not model_name in exclude:
                model_names.append(model_name)
                eval_data.append(json.load(open(model_dir_/"eval_data.json", "r"))["eval_data"])
                config = load_yaml(model_dir/"run_config.yml")
                split = config["seed"]
                splits.append(split)

    new_data = make_data_list(eval_data)

    if ref:
        ref_data = make_data_list(eval_data, reference=True)
        ref_splits = copy.deepcopy(splits)
        ref_models = [refname]*len(model_names)

        new_data += ref_data
        splits += ref_splits
        model_names += ref_models
    

    if not best_criterion is None:
        # for each split, loop through the model_names and for those that are the same, only keep the one that is the best according to the criterion
        assert len(best_criterion) == 2, f"Expected best_criterion to be a tuple of length 2, but got {best_criterion}."
        assert best_criterion[0] in ["energy_rmse", "grad_rmse"], f"Expected best_criterion[0] to be 'energy_rmse' or 'grad_rmse', but got {best_criterion[0]}."

        # get the best model for each split:
        best_models = []
        best_model_values = []
        best_new_data = []
        for split in set(splits):
            # condition on split:
            split_models = [model for model, s in zip(model_names, splits) if s == split]
            split_new_data = [data for data, s in zip(new_data, splits) if s == split]
            split_values = [data[best_criterion[1]]["val"][best_criterion[0]] for data in split_new_data]
            print(set(split_models))
            # now differentiate between different model names:
            for model in set(split_models):
                # condition on model:
                model_values = [value for value, m in zip(split_values, split_models) if m == model]
                model_new_data = [data for data, m in zip(split_new_data, split_models) if m == model]

                best_idx = np.argmin(model_values)
                best_models.append(copy.deepcopy(model))
                best_new_data.append(copy.deepcopy(model_new_data[best_idx]))

        # overwrite:
        new_data = best_new_data
        model_names = best_models
        print(model_names)

    compare(new_data, model_names, plotdir=parent_dir.parent, fontsize=fontsize, figsize=figsize, best=best, folds=folds)


def client():
    """
    Create a bar plot comparing model performance of models in subfolders of parent_dir.

    For each subfolder of the parent dir, check if it contains a file called 'eval_data.json'. Use the name of the subfolder as the model name. Then create a bar plot comparing the models in the parent of the parent dir.
    """
    parser = argparse.ArgumentParser(description="Compare the performance of different models on different datasets.")

    parser.add_argument("-d", "--parent_dir", type=str, help="The parent directory containing the models to compare.", default=Path.cwd()/"versions")
    parser.add_argument("--fontsize", type=int, default=14, help="The fontsize for the plot.")
    parser.add_argument("--figsize", type=int, nargs=2, default=(10,5), help="The figure size for the plot.")
    parser.add_argument("--vpaths", type=str, nargs="+", default=None, help="The paths to the versions to compare.")
    parser.add_argument("--exclude", type=str, nargs="+", default=[], help="The paths to the versions to compare.")
    parser.add_argument("--best_criterion", type=str, nargs=2, default=None, help="The criterion to use to select the best model for each split. First argument is the metric, second is the dataset type. If None, use all models.")
    parser.add_argument("--folds", action="store_true", help="If true, compare folds.")
    parser.add_argument("--best", action="store_true", help="If true, only compare the best model for each split.")
    parser.add_argument("--ref", action="store_true", help="If true, compare the reference model.")
    args = parser.parse_args()

    compare_versions(args.parent_dir, fontsize=args.fontsize, figsize=args.figsize, vpaths=args.vpaths, exclude=args.exclude, best_criterion=args.best_criterion, folds=args.folds, best=args.best, ref=args.ref)