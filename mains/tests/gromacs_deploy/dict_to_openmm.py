#%%
"""
Tests the creation of openmm system for various inputs. Also tests that radicals throw an error if allow_radiclas is False and HYP and DOP throw an error if the usual amber99sbildn is used.
"""
# %%
from grappa.ff import ForceField
from openmm.app import ForceField as openmm_ff
import openmm.unit
import grappa

#%%
model_path = "/hits/fast/mbm/seutelf/grappa/mains/runs/reworked_test/versions/17_good/best_model.pt"

collagen_ff = grappa.ff_utils.classical_ff.collagen_utility.get_collagen_forcefield()

ff = ForceField(model_path=model_path, classical_ff=collagen_ff)

# %%
from openmm.app import PDBFile
top = PDBFile("AG_rad/pep.pdb").topology
ff.allow_radicals = False
# this should fail:
try:
    sys = ff.createSystem(top)
    raise
except ValueError:
    pass

# this should work:
ff.allow_radicals = True
sys = ff.createSystem(top)

# %%
import json
with open('GrAPPa_input.json', "r") as f:
    data = json.load(f)

#%%
data["radicals"] = []
data["bonds"] += [[29, 35]]
ff.allow_radicals = False
ff.set_charge_model("heavy")
sys = ff.system_from_topology_dict(data)

# %%
with open('GrAPPa_input_tripelhelix.json', "r") as f:
    data = json.load(f)

# this should fail:
ff.allow_radicals = False
try:
    ff.classical_ff = openmm.app.ForceField("amber99sbildn.xml")
    sys = ff.system_from_topology_dict(data)
    raise
except ValueError:
    pass

# this should work:
ff.classical_ff = collagen_ff
sys = ff.system_from_topology_dict(data)

# %%

with open('GrAPPa_input_HAT.json', "r") as f:
    data = json.load(f)
ff.allow_radicals = False
# this should fail:
try:
    sys = ff.system_from_topology_dict(data)
    raise
except ValueError:
    pass

# this should work:
ff.allow_radicals = True
sys = ff.system_from_topology_dict(data)
# %%
