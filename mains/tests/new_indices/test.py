#%%
from grappa.run import run_utils
from grappa.ff_utils.create_graph import utils, tuple_indices
#%%
p = "/hits/fast/mbm/seutelf/data/datasets/old_PDBDatasets/spice/amber99sbildn_amber99sbildn_dgl.bin"
ds, _ = run_utils.get_data([p], 10)
# %%
import numpy as np
g = ds[0][1]
atoms = g.nodes["n1"].data["atomic_number"].numpy()[:,0].astype(np.int64).tolist()
bonds = g.nodes["n2"].data["idxs"].numpy().astype(np.int64).tolist()
bonds_ = [tuple(b) for b in bonds]
bonds = []
for b in bonds_:
    if (b[1], b[0]) not in bonds:
        bonds.append(b)

residues = ["ALA" for _ in range(len(atoms))]
atom_names = ["CA" for _ in range(len(atoms))]
#%%
mol = utils.bonds_to_rdkit_graph(bond_indices=bonds, atomic_numbers=atoms, residues=residues, atom_names=atom_names)
#%%
idxs_new = tuple_indices.get_indices(mol, reduce_symmetry=False)
for lvl in ["n2", "n3", "n4", "n4_improper"]:
    print(lvl)
    idxs_openff = g.nodes[lvl].data["idxs"].numpy().astype(np.int64)
    if lvl == "n4_improper":
        idxs_openff = idxs_openff[:, [0, 2, 1, 3]]
    new = idxs_new[lvl]
    new = set([tuple(n) for n in new])
    idxs_openff = set([tuple(n) for n in idxs_openff])
    if new != idxs_openff:
        print("not equal, length new: %s, length openff: %s" % (len(new), len(idxs_openff)))
        print("not in openff, in new: ", len(list(new - idxs_openff)), sorted(list(new - idxs_openff)))
        print("not in new, in openff: ", len(list(idxs_openff - new)), sorted(list(idxs_openff - new)))
        # print(*idxs_openff, sep="\n")
        print("It seems like openff ignored some impropers, e.g. 0 is never part of it.")
        print("Since all openff tuples are contained in the new tuples, it is fine.")

# %%
torsion_idxs = [tuple(t) for t in idxs_new["n4"]]
improper_idxs = [tuple(t) for t in idxs_new["n4_improper"]]

both = set(improper_idxs).intersection(set(torsion_idxs))
print("num proper: ", len(torsion_idxs))
print("num improper: ", len(improper_idxs))
print("impropers that are also torsions: ", len(both), sorted(list(both)))
# %%
