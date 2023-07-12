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
print(dihedral(x1, x2, x3, x4))
print(dihedral(x4, x2, x3, x1))
# %%
