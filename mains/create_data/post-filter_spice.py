#%%
import sys, os

from grappa.constants import DEFAULTBASEPATH

from pathlib import Path

spicedgl_path = DEFAULTBASEPATH / Path("spice_openff_full/charge_default_ff_gaff-2_11_60_dgl.bin")
# %%
import dgl
import torch

ds, _ = dgl.load_graphs(str(spicedgl_path))
num_confs = sum(map(lambda g: g.nodes["g"].data["u_qm"].shape[1], ds))
print(num_confs, len(ds))

# %%
def filter_confs(g, max_energy=65, max_force=300):
    energies = g.nodes["g"].data["u_ref"][0]
    energies -= energies.min()

    forces = g.nodes["n1"].data["grad_ref"].transpose(0,1)
    forces = torch.abs(forces)
    forces = torch.max(forces, dim=-1).values
    forces = torch.max(forces, dim=-1).values

    mask = (energies < max_energy) & (forces < max_force)

    if sum(mask) == 0:
        return None

    for key in g.nodes["g"].data.keys():
        g.nodes["g"].data[key] = g.nodes["g"].data[key][:,mask]
    for key in g.nodes["n1"].data.keys():
        if not key in ['residue', 'is_radical', 'q_ref', 'formal_charge', 'mass', 'in_ring', 'atomic_number']:
            g.nodes["n1"].data[key] = g.nodes["n1"].data[key][:,mask]
    
    return g

# %%
ds[:] = [g for g in map(filter_confs, ds) if not g is None]
# %%
num_confs = sum(map(lambda g: g.nodes["g"].data["u_qm"].shape[1], ds))
print(num_confs, len(ds))
# %%
store_path = DEFAULTBASEPATH / Path("spice_openff_full/charge_default_ff_gaff-2_11_forcefiltered_60_dgl.bin")

dgl.save_graphs(str(store_path), ds)
# %%
