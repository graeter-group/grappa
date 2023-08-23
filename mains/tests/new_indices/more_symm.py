#%%
from grappa.models.geometry import dihedral

import torch
pos = torch.randn(4, 3)
x1, x2, x3, x4 = pos
x1.shape
# %%
# permute the total order:
print(dihedral(x1, x2, x3, x4))
print(dihedral(x4, x3, x2, x1))
# %%
# permute outer atoms:
print(dihedral(x1, x2, x3, x4))
print(dihedral(x4, x2, x3, x1))
#%%
# permute the bonds:
print(dihedral(x1, x2, x3, x4))
print(dihedral(x2, x1, x4, x3))
# %%
# cyclic permutation keeping atom 2 fixed:
print(dihedral(x1, x2, x3, x4))
print(dihedral(x3, x2, x4, x1))
# %%
# cyclic permutation keeping atom 3 fixed:
print(dihedral(x1, x2, x3, x4))
print(dihedral(x4, x1, x3, x2))
# %%

def get_ab(x1, x2, x3, x4):
    x12 = x2 - x1
    x23 = x3 - x2
    x43 = x4 - x3

    a = torch.cross(x12, x23)
    b = torch.cross(x23, x43)
    return a, b

def get_cosine(x1, x2, x3, x4):
    a, b = get_ab(x1, x2, x3, x4)

    a_norm = torch.norm(a, dim=-1)
    b_norm = torch.norm(b, dim=-1)

    a_normed = a / a_norm
    b_normed = b / b_norm

    cosine = torch.sum(a_normed * b_normed)
    return cosine
#%%
print(get_ab(x1, x2, x3, x4))
print(get_ab(x4, x2, x3, x1))
# %%
