#%%
import openmm
from openmm.app import ForceField
from grappa.ff_utils.SysWriter import SysWriter
import json
from grappa.ff_utils.create_graph.utils import openmm2rdkit_graph
from grappa.ff_utils.create_graph.tuple_indices import get_indices
#%%
with open("in.json", "r") as f:
    in_dict = json.load(f)

#%%
writer = SysWriter.from_dict(in_dict, allow_radicals=True, classical_ff=ForceField("amber99sbildn.xml"))
# %%
sys = writer.sys
topology = writer.top
rdmol = openmm2rdkit_graph(topology)
# %%
writer.init_graph()
g = writer.graph

# %%
idxs_top = get_indices(rdmol)
idxs_top = {lvl : len(idxs_top[lvl]) for lvl in ["n2", "n3", "n4"]}
# %%
idxs_ff = {lvl : len(g.nodes[lvl].data["idxs"]) if "idxs" in g.nodes[lvl].data.keys() else 0 for lvl in ["n2", "n3", "n4"]}
# %%
print(idxs_ff)
# %%
print(idxs_top)
# %%
# %%
