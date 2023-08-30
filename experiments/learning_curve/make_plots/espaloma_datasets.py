#%%
from grappa.constants import DS_PATHS
from grappa.PDBData.PDBDataset import PDBDataset, SplittedDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from grappa.run.run_utils import load_yaml
from grappa.models.deploy import model_from_version
from grappa.ff import ForceField
import json
import os
from grappa.ff_utils.classical_ff.collagen_utility import get_mod_amber99sbildn

#%%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--vpath", '-v', help="Name of version folder", type=str, default="lc/lc_espaloma_new")
parser.add_argument("--ignore_ns", '-i', help="Ignore these n_mols", nargs='+', type=int, default=[])
parser.add_argument("--overwrite", "-o", action="store_true", default=False)
args = parser.parse_args()
grappa_vpath = f'/hits/fast/mbm/seutelf/grappa/mains/runs/{args.vpath}/versions'

foldername = Path(grappa_vpath).parent.name
os.makedirs(foldername, exist_ok=True)

n_max = None


calc_deviations = args.overwrite or not (Path(f"{foldername}/lc_data.json").exists() and Path(f"{foldername}/lc_energy_data.json").exists())

if calc_deviations:


    n_folds = 10 # used for checking for errors

    spice_monomers = PDBDataset.load_npz(DS_PATHS['spice_monomers'], n_max=n_max, info=False)

    spice_dipeptides_all = PDBDataset.load_npz(DS_PATHS['spice_qca'], n_max=n_max, info=False)

    spice_pubchem = PDBDataset.load_npz(DS_PATHS['spice_pubchem'], n_max=n_max, info=False)



    datasets = [spice_monomers, spice_dipeptides_all, spice_pubchem]
    ds_names = ["Spice Monomers", "Spice Dipeptides", "Spice Pubchem"]


    #%%




    data = {}
    data_energy = {}

    first = True
    for ds_name, ds in zip(ds_names, datasets):
        
        data[ds_name] = {}
        data_energy[ds_name] = {}
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
            # ff = ForceField(
            #     model_path=p / 'best_model.pt',
            #     classical_ff=get_mod_amber99sbildn(),
            #     allow_radicals=True)

            model = model_from_version(p)

            ds_te.calc_ff_data(model=model, suffix="_grappa", allow_radicals=True)
            eval_data, _ = ds_te.evaluate(plotpath=None, suffix="_grappa")
            force_rmse = eval_data['grad_rmse']
            energy_rmse = eval_data['energy_rmse']

            if n_mols in data[ds_name].keys():
                data[ds_name][n_mols].append(force_rmse)
                data_energy[ds_name][n_mols].append(energy_rmse)
            else:
                data[ds_name][n_mols] = [force_rmse]
                data_energy[ds_name][n_mols] = [energy_rmse]

        ds_name_ = ds_name.replace(" ", "_")
        ds_name_ = ds_name.replace("/", "-")
        

        with open(f"{foldername}/{ds_name_}_lc_data.json", "w") as f:
            json.dump(data[ds_name], f, indent=4)
        with open(f"{foldername}/{ds_name_}_lc_energy_data.json", "w") as f:
            json.dump(data_energy[ds_name], f, indent=4)

    with open(f"{foldername}/lc_data.json", "w") as f:
        json.dump(data, f, indent=4)
    
    with open(f"{foldername}/lc_energy_data.json", "w") as f:
        json.dump(data_energy, f, indent=4)

#%%

esp_errors = {
    "Spice Monomers": 6.05 * np.sqrt(3),
    "Spice Dipeptides": 7.85 * np.sqrt(3),
    "Spice Pubchem": 6.87 * np.sqrt(3),
}

from lc_plot import lc_plot

with open(f"{foldername}/lc_data.json", "r") as f:
    data = json.load(f)

lc_plot(data, title="Learning Curve", fontname="Arial", ylabel="Force RMSE [kcal/mol/Å]", show=False, plotpath=f"{foldername}", fit=False, logx=True, logy=False, ignore_n_worst=0, connect_dots=True, ignore_ns=args.ignore_ns, hlines=esp_errors)


esp_errors_energies = {
    "Spice Monomers": 1.68,
    "Spice Dipeptides": 3.15,
    "Spice Pubchem": 2.52,
}


with open(f"{foldername}/lc_energy_data.json", "r") as f:
    data = json.load(f)

lc_plot(data, title="Learning Curve", fontname="Arial", ylabel="Energy RMSE [kcal/mol]", show=False, plotpath=f"{foldername}", fit=False, logx=True, logy=False, ignore_n_worst=0, connect_dots=True, ignore_ns=args.ignore_ns, hlines=esp_errors_energies, suffix="_energy")