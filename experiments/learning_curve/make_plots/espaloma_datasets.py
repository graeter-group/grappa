#%%
from grappa.constants import DS_PATHS
from grappa.PDBData.PDBDataset import PDBDataset, SplittedDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from grappa.run.run_utils import load_yaml
from grappa.models.deploy import model_from_path
from grappa.ff import ForceField
import json
import os
from grappa.ff_utils.classical_ff.collagen_utility import get_mod_amber99sbildn
#%%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--vpath", '-v', help="Name of version folder", type=str, default="lc/lc_grappa_new")
parser.add_argument("--no_tripeptides", "-t", action="store_true", default=False)
parser.add_argument("--overwrite", "-o", action="store_true", default=False)
args = parser.parse_args()
grappa_vpath = f'/hits/fast/mbm/seutelf/grappa/mains/runs/{args.vpath}/versions'

foldername = Path(grappa_vpath).parent.name
os.makedirs(foldername, exist_ok=True)

n_max = None


calc_deviations = args.overwrite or not Path(f"{foldername}/lc_data.json").exists()

if calc_deviations:


    n_folds = 10 # used for checking for errors

    spice_grappa = PDBDataset.load_npz(DS_PATHS['spice'], n_max=n_max, info=False)
    collagen = PDBDataset.load_npz(DS_PATHS['collagen'], n_max=n_max, info=False)
    radical_AAs = PDBDataset.load_npz(DS_PATHS['radical_AAs'], n_max=n_max, info=False)
    radical_dipeptides = PDBDataset.load_npz(DS_PATHS['radical_dipeptides'], n_max=n_max, info=False)

    datasets = [radical_AAs, radical_dipeptides, spice_grappa, collagen]
    ds_names = ["Radical AAs", "Radical Dipeptides", "Spice Dipeptides", "HYP/DOP Dipeptides"]
    if not args.no_tripeptides:
        tripeptides = PDBDataset.load_npz(DS_PATHS['tripeptides'], n_max=n_max, info=False)
        datasets.append(tripeptides)
        ds_names.append("Tripeptides")

    #%%




    data = {}

    first = True
    for ds_name, ds in zip(ds_names, datasets):
        
        data[ds_name] = {}
        # data[ds_name+"_amber"] = []

        paths = list(Path(grappa_vpath).glob('*'))


        print(f"\nLoading {len(paths)} models for {ds_name}\n")

        for j, p in enumerate(paths):
            print(f"Loading model {j+1}/{len(paths)}")

            run_config = load_yaml(p / 'run_config.yml')

            # get an estimate for the maximum number of training molecules
            if first:
                ds_paths = run_config['ds_path']
                full_datasets = [PDBDataset.load_npz(ds_path, n_max=n_max, info=False) for ds_path in ds_paths]
                ds_splitter = SplittedDataset.load_from_names(str(p/Path("split")), datasets=full_datasets)
                train_datasets = ds_splitter.get_splitted_datasets(full_datasets, ds_type="tr")
                max_mols = 0
                for ds_tr in train_datasets:
                    max_mols += len(ds_tr)
                first = False

            n_mols = run_config['mols']

            seed = run_config['seed']

            ds_splitter = SplittedDataset.load_from_names(str(p/Path("split")), datasets=[ds])

            [ds_te] = ds_splitter.get_splitted_datasets([ds], ds_type="te")

            if len(ds_te) == 0:
                print(f"skipping {p} because test set is empty for {p.name}.")
                continue


            if n_mols is None:
                n_mols = max_mols
            if n_mols > max_mols:
                n_mols = max_mols


            # if n_mols == max_mols:
            #     eval_data_amber, _ = ds_te.evaluate(plotpath=None, suffix="_total_ref")

            #     amber_force_rmse = eval_data_amber['grad_rmse']
            #     data[ds_name+"_amber"].append(amber_force_rmse)


            # evaluate on test set:
            ff = ForceField(
                model_path=p / 'best_model.pt',
                classical_ff=get_mod_amber99sbildn(),
                allow_radicals=True)

            ds_te.calc_ff_data(ff, suffix="_grappa", allow_radicals=True)
            eval_data, _ = ds_te.evaluate(plotpath=None, suffix="_grappa")
            force_rmse = eval_data['grad_rmse']

            if n_mols in data[ds_name].keys():
                data[ds_name][n_mols].append(force_rmse)
            else:
                data[ds_name][n_mols] = [force_rmse]

        ds_name_ = ds_name.replace(" ", "_")
        ds_name_ = ds_name.replace("/", "-")
        

        with open(f"{foldername}/{ds_name_}_lc_data.json", "w") as f:
            json.dump(data[ds_name], f, indent=4)

    with open(f"{foldername}/lc_data.json", "w") as f:
        json.dump(data, f, indent=4)

#%%

from lc_plot import lc_plot

with open(f"{foldername}/lc_data.json", "r") as f:
    data = json.load(f)

lc_plot(data, title="Learning Curve", fontname="Arial", ylabel="Force RMSE [kcal/mol/Å]", show=False, plotpath=f"{foldername}", fit=False, logx=True, logy=True, ignore_n_worst=1, connect_dots=True)