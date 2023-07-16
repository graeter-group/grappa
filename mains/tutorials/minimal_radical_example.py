#%%
# imports:
from grappa.ff import ForceField
from grappa.ff_utils.classical_ff.collagen_utility import get_collagen_forcefield
import openmm.unit
from grappa.constants import TopologyDict, ParamDict


# load data describing a radical:
import json
with open('GrAPPa_input_HAT.json', "r") as f:
    data: TopologyDict = json.load(f)

# path to the grappa force field:
mpath = "/hits/fast/mbm/seutelf/grappa/mains/runs/stored_models/tutorial/best_model.pt"

# the classical force field grappa is built upon:
# from this we get the improper indices (these are not unique)
classical_ff = get_collagen_forcefield()

# initialize the force field:
ff = ForceField(model_path=mpath, classical_ff=classical_ff)

# set angle to degrees:
ff.units["angle"] = openmm.unit.degree

# allow radicals:
ff.allow_radicals = True
# %%
# predict parameters
params: ParamDict = ff.params_from_topology_dict(data)

#%%
# investigate the output:

print("dict keys and mean values:\n")
print(*[(k, v.mean()) for (k,v) in zip(params.keys(), params.values())], sep="\n")

print("\n\nsome equiluilibrium angles:")
print(params["angle_eq"][:5])
print("\n\nsome charges:")
print(params["atom_q"][:5])
print()

#%%
# get information on the units by printing the forcefield:
print(ff)
# %%
