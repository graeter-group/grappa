#%%
# imports:
from grappa.ff import ForceField
from grappa.constants import TopologyDict, ParamDict


# load data describing a radical:
import json
with open('GrAPPa_input_HAT.json', "r") as f:
    data: TopologyDict = json.load(f)

# initialize the forcefield from a tag:
ff = ForceField.from_tag("example")

# %%
# predict parameters
params: ParamDict = ff.params_from_topology_dict(data)

#%%
# investigate the output:

print("dict keys and mean values:\n")
print(*[(k, v.mean()) for (k,v) in zip(params.keys(), params.values())], sep="\n")

print("\n\nsome equililibrium angles:")
print(params["angle_eq"][:5])
print("\n\nsome charges:")
print(params["atom_q"][:5])
print()

#%%
# get information on the units by printing the forcefield:
print(ff)
# %%
