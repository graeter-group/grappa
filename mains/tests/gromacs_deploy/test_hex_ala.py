#%%
import json
with open('GrAPPa_input.json', "r") as f:
    data = json.load(f)

#%%
data["radicals"] = []
data["bonds"] += [[29, 35]]
# %%
from grappa.ff import ForceField
from openmm.app import ForceField as openmm_ff
import openmm.unit
import grappa

ff = ForceField(model=ForceField.get_zero_model())
# %%
sys = ff.system_from_topology_dict(data)
#%%
model_path = "/hits/fast/mbm/seutelf/grappa/mains/runs/reworked_test/versions/17_good/best_model.pt"

collagen_ff = grappa.ff_utils.classical_ff.collagen_utility.get_collagen_forcefield()

ff = ForceField(model_path=model_path, classical_ff=collagen_ff)
ff.units["angle"] = openmm.unit.degree

# %%
# %%
with open('GrAPPa_input_tripelhelix.json', "r") as f:
    data = json.load(f)

params = ff(data)
# %%
print(params["angle_eq"][:6])
# %%
# should fail:
try:
    with open('GrAPPa_input_HAT.json', "r") as f:
        data = json.load(f)
    params = ff(data)
    raise
except:
    pass

# should work:
ff.allow_radicals = True
params = ff(data)

# %%
# we did not provide a charge model, so the systen does not have integer charge:
q = params["atom_q"]
q.sum()
# %%
# now with a charge model:
ff.set_charge_model("heavy")
params = ff(data)
q = params["atom_q"]
q.sum()
# %%
