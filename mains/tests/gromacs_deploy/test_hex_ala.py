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

ff = ForceField(model=ForceField.get_zero_model())
# %%
ff(data)
#%%
mpath = "/hits/fast/mbm/seutelf/grappa/mains/runs/param_search/versions/5__dotted_512_6_6_full_ds/best_model.pt"
ff = ForceField(model_path=mpath)

# %%
params = ff(data)
# %%
with open('GrAPPa_input_tripelhelix.json', "r") as f:
    data = json.load(f)

params = ff(data)
# %%
with open('GrAPPa_input_HAT.json', "r") as f:
    data = json.load(f)

params = ff(data)
# %%
params["bond_k"]
# %%
