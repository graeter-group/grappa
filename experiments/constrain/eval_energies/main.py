#%%
from grappa.ff import ForceField
import openmm.unit
import openmm.app
from grappa.ff_utils.classical_ff.collagen_utility import get_amber99sbildn
from pathlib import Path
from openmm import unit
from grappa.PDBData.PDBDataset import PDBDataset
from grappa.constants import DS_PATHS
import json

#%%
ds_path = DS_PATHS["large_peptides"]

ds = PDBDataset.load_npz(ds_path, n_max=None)
# %%

LATEST = False

for i,p in [(0,1),(1,10),(2,100),(3,1000)]:
    vpath = f"/hits/fast/mbm/seutelf/grappa/mains/runs/constrain/versions/{i}_{p}"

    param_weight = Path(vpath).name.split("_")[-1]

    ff_ref = ForceField(classical_ff=get_amber99sbildn(), model=None)

    ff = ForceField(classical_ff=get_amber99sbildn(), model_path=Path(vpath)/"best_model.pt")

    if LATEST:
        ff = ForceField.from_tag("latest")



    ds.calc_ff_data(forcefield=ff, suffix="")


    eval_data, _ = ds.evaluate(plotpath=str(param_weight), by_element=True, by_residue=True, suffix="", name="Grappa", fontsize=16, fontname="Arial", refname="Amber 99SBildn")

    energy_rmse = eval_data['energy_rmse']
    force_rmse = eval_data['grad_rmse']

    if Path("energy_force_rmse.json").exists():
        with open("energy_force_rmse.json", "r") as f:
            all_data = json.load(f)
    else:
        all_data = {}


    all_data[param_weight] = [energy_rmse, force_rmse]

    with open("energy_force_rmse.json", "w") as f:
        json.dump(all_data, f)

#%%