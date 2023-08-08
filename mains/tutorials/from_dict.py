#%%
# load data dictionary:
import json
with open('input_data/GrAPPa_input_HAT.json', "r") as f:
    data = json.load(f)

from grappa.ff import ForceField
from grappa.ff_utils.classical_ff.collagen_utility import get_mod_amber99sbildn
import openmm.unit

# path to the grappa model the forcefield is supposed to use (this can also be done by tag):
mpath = "/hits/fast/mbm/seutelf/grappa/mains/runs/stored_models/example/best_model.pt"

# classical forcefield that grapp auses for the nonbonded interactions:
classical_ff = get_mod_amber99sbildn()

ff = ForceField(model_path=mpath, classical_ff=classical_ff)

# set angle to degrees:
ff.units["angle"] = openmm.unit.degree

# allow radicals:
ff.allow_radicals = True
# %%
# predict parameters
params = ff.params_from_topology_dict(data)
#%%
print("dict keys and mean values:\n")
print(*[(k, v.mean()) for (k,v) in zip(params.keys(), params.values())], sep="\n")

print("\n\nsome equiluilibrium angles:")
print(params["angle_eq"][:5])
print("\n\nsome charges:")
print(params["atom_q"][:5])
# %%
print("units:\n")
print(ff)



# %%
# for a triplehelix the parametrization takes longer (~15 seconds):
with open('input_data/GrAPPa_input_tripelhelix.json', "r") as f:
    data = json.load(f)

params = ff.params_from_topology_dict(data)
print("dict keys and mean values:\n")
print(*[(k, v.mean()) for (k,v) in zip(params.keys(), params.values())], sep="\n")

print(f"\n\nnum_atoms:\n{len(params['atom_q'])}")
# %%
