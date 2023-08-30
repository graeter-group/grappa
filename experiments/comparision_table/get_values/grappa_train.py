#%%
from grappa.constants import DS_PATHS
from grappa.PDBData.PDBDataset import PDBDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from grappa.run.run_utils import load_yaml
from grappa.models.deploy import model_from_path
from grappa.ff import ForceField
import json
from grappa.ff_utils.classical_ff.collagen_utility import get_mod_amber99sbildn
#%%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--vpath", '-v', help="Path to version folder", default="grappa_pep", type=str)
parser.add_argument("--tripeptides", "-t", help="Use tripeptides too", action="store_true")
args = parser.parse_args()
grappa_vpath = f'/hits/fast/mbm/seutelf/grappa/mains/runs/{args.vpath}/versions'


n_max = None

n_folds = 10 # used for checking for errors

spice_grappa = PDBDataset.load_npz(DS_PATHS['spice'], n_max=n_max, info=False)
collagen = PDBDataset.load_npz(DS_PATHS['collagen'], n_max=n_max, info=False)
radical_AAs = PDBDataset.load_npz(DS_PATHS['radical_AAs'], n_max=n_max, info=False)
radical_dipeptides = PDBDataset.load_npz(DS_PATHS['radical_dipeptides'], n_max=n_max, info=False)

datasets = [radical_AAs, radical_dipeptides, spice_grappa, collagen]
ds_names = ["Radical_AAs", "Radical_Dipeptides", "Spice_Grappa", "Collagen"]
if args.tripeptides:
    tripeptides = PDBDataset.load_npz(DS_PATHS['tripeptides'], n_max=n_max, info=False)
    datasets.append(tripeptides)
    ds_names.append("Tripeptides")

#%%

foldername = Path(grappa_vpath).parent.name

for ds_name, ds in zip(ds_names, datasets):
    
    with_radicals = "radical" in ds_name.lower()

    ds.calc_ff_data(get_mod_amber99sbildn(), suffix="_amber99", allow_radicals=True)
    eval_data, _ = ds.evaluate(plotpath=f"{foldername}/{ds_name}_plots/amber", suffix="_amber99", name="Amber ff99SBildn", refname="QM")
    amber_energy_rmse = eval_data['energy_rmse']
    amber_force_rmse = eval_data['grad_rmse']
    amber_force_components_rmse = eval_data['component_grad_rmse']
    amber_energy_mae = eval_data['energy_mae']
    amber_force_mae = eval_data['grad_mae']    

    energy_std = eval_data['energy_L2']
    grad_std = eval_data['grad_L2']
    n_mols = eval_data['n_mols']
    n_confs = eval_data['n_confs']


    paths = list(Path(grappa_vpath).glob('*'))


    # this stores the best model in terms of val energy rmse for each fold
    # filtering is okay as long as only val is used, not test set!
    ds_criterion = "spice"
    metric_criterion = "energy_rmse"
    best_fold_data = {} # keys: fold, out: [path, metric_value]
    folds = []

    for j, p in enumerate(paths):
        run_config = load_yaml(p / 'run_config.yml')
        fold = run_config['ds_split_names']

        assert not fold is None, "Fold is None, this should not happen"

        folds.append(fold)
        try:
            with open(p/Path("eval_data.json"), "r") as f:
                data = json.load(f)
                metric = data["eval_data"][f"{ds_criterion}_val"][metric_criterion]
        except:
            metric = float("inf")

        if fold in best_fold_data.keys():
            if metric < best_fold_data[fold][1]:
                best_fold_data[fold] = [p, metric]
        else:
            best_fold_data[fold] = [p, metric]
        

    assert len(list(best_fold_data.keys())) == n_folds, f"Not all folds are present in best_fold_data: {list(best_fold_data.keys())}"

    best_paths = [best_fold_data[fold][0] for fold in best_fold_data.keys()]

    full_test_ds = PDBDataset(info=False)

    print(f"\nLoading {len(best_paths)} models for {ds_name}\n")

    for j, p in enumerate(best_paths):
        print(f"Loading model {j+1}/{len(best_paths)}")
        run_config = load_yaml(p / 'run_config.yml')
        fold = run_config['ds_split_names']

        assert not fold is None, "Fold is None, this should not happen"

        ds_tr, ds_vl, sub_ds = ds.split_by_names(fold)

        ff = ForceField(
            model_path=p / 'best_model.pt',
            classical_ff=get_mod_amber99sbildn(),
            allow_radicals=True)

        sub_ds.calc_ff_data(ff, suffix="")
        
        full_test_ds += sub_ds

    # assert len(full_test_ds) == len(ds), f"Length of full_test_ds ({len(full_test_ds)}) does not match length of ds ({len(ds)})"

    eval_data, _ = full_test_ds.evaluate(plotpath=f"{foldername}/{ds_name}_plots", by_element=True, by_residue=True, suffix="", name="Grappa", compare_name="Amber ff99SBildn", compare_suffix="_amber99", fontsize=16, fontname="Arial", refname="QM", radicals=with_radicals)

    energy_rmse = eval_data['energy_rmse']
    force_rmse = eval_data['grad_rmse']
    force_components_rmse = eval_data['component_grad_rmse']
    energy_mae = eval_data['energy_mae']
    force_mae = eval_data['grad_mae']

    with open(f"{foldername}/{ds_name}_data.json", "w") as f:
        data_dict = {
            "energy_rmse": energy_rmse,
            "force_rmse": force_rmse,
            "force_components_rmse": force_components_rmse,
            "amber_energy_rmse": amber_energy_rmse,
            "amber_force_rmse": amber_force_rmse,
            "amber_force_components_rmse": amber_force_components_rmse,
            "energy_std": energy_std,
            "grad_std": grad_std,
            "n_mols": n_mols,
            "n_confs": n_confs,
            "energy_mae": energy_mae,
            "force_mae": force_mae,
        }
        json.dump(data_dict, f, indent=4)
    # %%
