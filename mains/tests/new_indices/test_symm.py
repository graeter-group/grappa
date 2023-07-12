#%%
from grappa.run import run_utils
from grappa.ff_utils.create_graph import utils, tuple_indices
#%%
p = "/hits/fast/mbm/seutelf/data/datasets/old_PDBDatasets/spice/amber99sbildn_amber99sbildn_dgl.bin"
ds, _ = run_utils.get_data([p], 10)
# %%
import numpy as np
g = ds[0][1]


# %%
from grappa.models.geometry import dihedral

improper_idxs = g.nodes["n4_improper"].data["idxs"]
improper_idxs = improper_idxs[:, [0, 2, 1, 3]][0] # apply our new symmetry
pos = g.nodes["n1"].data["xyz"][:, 0, :]
pos = pos[improper_idxs]
pos.shape
import torch
pos = torch.randn(4, 3)
x1, x2, x3, x4 = pos
x1.shape
#%%
# all permutations:
import itertools
perms = [np.array(perm) for perm in itertools.permutations([0, 1, 2, 3])]
for permutation in perms:
    print(dihedral(*pos[permutation]))
# %%

# all permutations that keep atom number 1 fixed:
perms = [np.array(perm) for perm in itertools.permutations([0, 2, 3])]
for permutation in perms:
    perm = np.array([permutation[0], 1, *permutation[1:]])
    print(dihedral(*pos[perm]))
# %%
# all permutations that keep atom number 2 fixed:
perms = [np.array(perm) for perm in itertools.permutations([0, 1, 3])]
for permutation in perms:
    perm = np.array([permutation[0], permutation[1], 2, permutation[2]])
    print(dihedral(*pos[perm]))
# %%
# all cyclic permutations that keep atom number 1 fixed:
perms = [(0,1,2,3), (3,1,0,2), (2,1,3,0)]
for permutation in perms:
    perm = np.array(permutation)
    print(dihedral(*pos[perm]))
# %%
# all cyclic permutations that keep atom number 2 fixed:
perms = [(0,1,2,3), (3,0,2,1), (1,3,2,0)]
for permutation in perms:
    perm = np.array(permutation)
    print(dihedral(*pos[perm]))
# %%
# permuting the total order:
perms = [(0,1,2,3), (3,2,1,0)]
for permutation in perms:
    perm = np.array(permutation)
    print(dihedral(*pos[perm]))
# %%
# permuting both bonds:
perms = [(0,1,2,3), (2,3,0,1)]
for permutation in perms:
    perm = np.array(permutation)
    print(dihedral(*pos[perm]))

# %%
# all cyclic permutations:
perms = [(0,1,2,3), (3,0,1,2), (2,3,0,1), (1,2,3,0)]
for permutation in perms:
    perm = np.array(permutation)
    print(dihedral(*pos[perm]))
# %%
