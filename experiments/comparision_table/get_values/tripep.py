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
args = parser.parse_args()
grappa_vpath = f'/hits/fast/mbm/seutelf/grappa/mains/runs/{args.vpath}/versions'


n_max = None

tripep_ds = PDBDataset.load_npz(DS_PATHS['tripeptides'], n_max=n_max)
foldername = Path(grappa_vpath).parent.name

tripep_ds.calc_ff_data(get_mod_amber99sbildn(), suffix="_amber99")
eval_data, _ = tripep_ds.evaluate(plotpath=f"{foldername}/tripep_plots/amber", suffix="_amber99", name="Amber ff99SBildn", refname="QM")
amber_energy_rmse = eval_data['energy_rmse']
amber_force_rmse = eval_data['grad_rmse']
amber_force_components_rmse = eval_data['component_grad_rmse']
amber_energy_mae = eval_data['energy_mae']
amber_force_mae = eval_data['grad_mae']

energy_std = eval_data['energy_L2']
grad_std = eval_data['grad_L2']
n_mols = eval_data['n_mols']
n_confs = eval_data['n_confs']

# spice_grappa = PDBDataset.load_npz(DS_PATHS['spice'], n_max=n_max)
# %%


energy_rmse = []
force_rmse = []
force_components_rmse = []
energy_mae = []
force_mae = []


first = True
paths = list(Path(grappa_vpath).glob('*'))
for i, p in enumerate(paths):
    print(f"Loading model {i+1}/{len(paths)}")
    run_config = load_yaml(p / 'run_config.yml')
    fold = run_config['ds_split_names']

    if not fold is None:
        with open(fold, 'r') as f:
            fold = json.load(f)


    ff = ForceField(
        model_path=p / 'best_model.pt',
        classical_ff=get_mod_amber99sbildn(),
        allow_radicals=False)

    tripep_ds.calc_ff_data(ff, suffix="")
    
    eval_data, _ = tripep_ds.evaluate(plotpath=None, suffix="")

    # plot only for the first model
    if first:
        tripep_ds.evaluate(plotpath=f"{foldername}/tripep_plots", by_element=True, by_residue=True, suffix="", name="Grappa", compare_name="Amber ff99SBildn", compare_suffix="_amber99", fontsize=16, fontname="Arial", refname="QM")

        tripep_ds.eval_params(plotpath=f"{foldername}/tripep_plots", ff_name="Grappa", ff=ff, fontname="Arial", ref_name="Amber ff99SBildn")

        first = False



    energy_rmse.append(eval_data['energy_rmse'])
    force_rmse.append(eval_data['grad_rmse'])
    force_components_rmse.append(eval_data['component_grad_rmse'])
    energy_mae.append(eval_data['energy_mae'])
    force_mae.append(eval_data['grad_mae'])


#%%
# calc the statistics of the grappa metrics:
energy_rmse_ = np.array(energy_rmse)
force_rmse_ = np.array(force_rmse)
force_components_rmse_ = np.array(force_components_rmse)

energy_rmse_mean = float(np.mean(energy_rmse_, axis=0))
energy_rmse_std = float(np.std(energy_rmse_, axis=0))

force_rmse_mean = float(np.mean(force_rmse_, axis=0))
force_rmse_std = float(np.std(force_rmse_, axis=0))

force_components_rmse_mean = float(np.mean(force_components_rmse_, axis=0))
force_components_rmse_std = float(np.std(force_components_rmse_, axis=0))

force_mae_mean = float(np.mean(force_mae, axis=0))
force_mae_std = float(np.std(force_mae, axis=0))

energy_mae_mean = float(np.mean(energy_mae, axis=0))
energy_mae_std = float(np.std(energy_mae, axis=0))

# %%
with open(f"{foldername}/tripep_data.json", "w") as f:
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
        "energy_rmse_mean": energy_rmse_mean,
        "energy_rmse_std": energy_rmse_std,
        "force_rmse_mean": force_rmse_mean,
        "force_rmse_std": force_rmse_std,
        "force_components_rmse_mean": force_components_rmse_mean,
        "force_components_rmse_std": force_components_rmse_std,
        "energy_mae_mean": energy_mae_mean,
        "energy_mae_std": energy_mae_std,
        "force_mae_mean": force_mae_mean,
        "force_mae_std": force_mae_std,
        "amber_energy_mae": amber_energy_mae,
        "amber_force_mae": amber_force_mae,
    }
    json.dump(data_dict, f, indent=4)
# %%
