#%%
from grappa.utils import tuple_indices, rdkit_utils
import numpy as np

N = 1000
atoms = list(range(N))

# unique tuples of N random integers where idx0 < idx1:
bonds = [(np.random.randint(0, N), np.random.randint(0, N)) for _ in range(N)]
bonds = [tuple(sorted(bond)) for bond in bonds]
bonds = list(set(bonds))
bonds = [bond for bond in bonds if not bond[0] == bond[1]] 
len(bonds)

# %%
from time import time

t0 = time()

d = tuple_indices.get_idx_tuples(bonds)

t1 = time()

mol = rdkit_utils.rdkit_graph_from_bonds(bonds)
d0 = {
    'bonds': bonds,
    'angles': rdkit_utils.angle_indices(mol),
    'propers': rdkit_utils.torsion_indices(mol),
}

t2 = time()

print(f"own method: {t1-t0:.1e}")
print(f"rdkit method: {t2-t1:.1e}")
# %%

# convert to tuples:
d0['angles'] = d0['angles'].tolist()
d0['propers'] = d0['propers'].tolist()

# sort the rdkit tuples:
for i,angle in enumerate(d0['angles']):
    (a0, a1, a2) = angle
    d0['angles'][i] = (a0, a1, a2) if a0 < a2 else (a2, a1, a0)

for i,proper in enumerate(d0['propers']):
    (a0, a1, a2, a3) = proper
    d0['propers'][i] = (a0, a1, a2, a3) if a0 < a3 else (a3, a2, a1, a0)
# %%
d['angles'] = [tuple(angle) for angle in d['angles']]
d['propers'] = [tuple(proper) for proper in d['propers']]
# %%
assert set(d['angles']) == set(d0['angles'])
assert set(d['propers']) == set(d0['propers'])

# %%
