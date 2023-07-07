#%%
# load data dictionary:
import json
with open('GrAPPa_input_HAT.json', "r") as f:
    data = json.load(f)

from grappa.ff import ForceField
import openmm.unit

# path to the grappa model the forcefield is supposed to use:
mpath = "/hits/fast/mbm/seutelf/grappa/mains/runs/param_search/versions/5__dotted_512_6_6_full_ds/best_model.pt"
ff = ForceField(model_path=mpath)

# set angle to degrees:
ff.units["angle"] = openmm.unit.degree
# %%
# predict parameters
params = ff.params_from_topology_dict(data)
print("dict keys and mean values:\n")
print(*[(k, v.mean()) for (k,v) in zip(params.keys(), params.values())], sep="\n")

print("\n\nequiluilibrium angles:")
print(params["angle_eq"])
print("\n\ncharges:")
print(params["atom_q"])
# %%
print("units:\n")
print(ff)



# %%
# for a triplehelix the parametrization takes longer (~15 seconds):
with open('GrAPPa_input_tripelhelix.json', "r") as f:
    data = json.load(f)

params = ff.params_from_topology_dict(data)
print("dict keys and mean values:\n")
print(*[(k, v.mean()) for (k,v) in zip(params.keys(), params.values())], sep="\n")

print(f"\n\nnum_atoms:\n{len(params['atom_q'])}")
# %%
