#%%

# imports:
from grappa.ff import ForceField
import openmm.unit
import openmm.app
from grappa.ff_utils.classical_ff.collagen_utility import get_amber99sbildn
from pathlib import Path
from openmm import unit

# initialize the force field from a tag:
# ff = ForceField.from_tag("latest")

LATEST = True


vpath = "/hits/fast/mbm/seutelf/grappa/mains/runs/constrain/versions/3_1000"

param_weight = Path(vpath).name.split("_")[-1]

ff_ref = ForceField(classical_ff=get_amber99sbildn(), model=None)

ff = ForceField(classical_ff=get_amber99sbildn(), model_path=Path(vpath)/"best_model.pt")

if LATEST:
    ff = ForceField.from_tag("latest")


ff.units["angle"] = unit.degree
ff.units["distance"] = unit.angstrom
ff.units["energy"] = unit.kilocalorie_per_mole

ff_ref.units["angle"] = unit.degree
ff_ref.units["distance"] = unit.angstrom
ff_ref.units["energy"] = unit.kilocalorie_per_mole

# ff = openmm.app.ForceField('amber99sbildn.xml', 'tip3p.xml') # uncomment for comparison
#%%

# load example data:
from openmm.app import PDBFile, Modeller

# load pdb and add hydrogens:
pdb = PDBFile("1aki.pdb")
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addHydrogens(openmm.app.ForceField('amber99sbildn.xml', 'tip3p.xml'))
# remove water:
modeller.deleteWater()
top = modeller.getTopology()
positions = modeller.getPositions()
#%%
# get a system:
param_dict = ff.params_from_topology_dict(top)
param_dict_ref = ff_ref.params_from_topology_dict(top)
# %%
from grappa.PDBData.utils.utils import eval_params
eval_params(param_dict, param_dict_ref, plotpath=str(param_weight) if not LATEST else "latest", fontsize=16, ff_name="Grappa", ref_name="Amber ff99SBildn", collagen=False, fontname="Arial", figsize=6)
#%%

print(f"Num Residues: {len(list(top.residues()))}")
print(f"Num Atoms: {len(list(top.atoms()))}")

with open("aki_params.txt", "w") as f:
    f.write(f"Num Residues: {len(list(top.residues()))}\n")
    f.write(f"Num Atoms: {len(list(top.atoms()))}\n")
# %%
