#%%
from grappa.models.geometry import dihedral

import torch
pos = torch.randn(4, 3)
x1, x2, x3, x4 = pos
# %%
# all permutations:
import itertools
import numpy as np
perms = [np.array(perm) for perm in itertools.permutations([0, 1, 2, 3])]
symm = []
antisymm = []
dih_sym = []
dih_antisym = []
by_pi = []
for p in perms:
    if torch.allclose(dihedral(*pos[p]), dihedral(x1, x2, x3, x4), atol=1e-3):
        symm.append(p)
        dih_sym.append(dihedral(*pos[p]))
    elif torch.allclose(dihedral(*pos[p]), -dihedral(x1, x2, x3, x4), atol=1e-3):
        antisymm.append(p)
        dih_antisym.append(dihedral(*pos[p]))
    # check whether adding or subtracting pi makes the angles the same:
    elif torch.allclose(dihedral(*pos[p]) + np.pi, dihedral(x1, x2, x3, x4), atol=1e-3):
        by_pi.append(p)
    elif torch.allclose(dihedral(*pos[p]) - np.pi, dihedral(x1, x2, x3, x4), atol=1e-3):
        by_pi.append(p)
# %%
print(len(perms))
print()
print(len(symm), *symm, sep="\n")
print()
print(len(antisymm), *antisymm, sep="\n")
# %%
print(*dih_sym, sep="\n")
print(*dih_antisym, sep="\n")
# %%
print(len(by_pi), *by_pi, sep="\n")
# %%
