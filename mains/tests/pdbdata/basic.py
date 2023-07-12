#%%

from pathlib import Path
import tempfile
import dgl
import numpy as np

from grappa.PDBData.PDBDataset import PDBDataset

#%%
spicepath = Path(__file__).parent
dspath = Path(spicepath)/Path("small_spice.hdf5")
ds = PDBDataset.from_spice(dspath, skip_errs=True)

#%%
print(*ds[3].pdb)
# %%
ds.filter_validity()
len(ds)
# %%
ds.filter_confs()
len(ds)
#%%
# check saving and loading:

with tempfile.TemporaryDirectory() as tmpdirname:
    ds.save_npz(tmpdirname, overwrite=True)
    ds2 = PDBDataset.load_npz(tmpdirname)

assert len(ds) == len(ds2), f"lengths are not the same: {len(ds)} vs {len(ds2)}"

assert set([mol.xyz.shape for mol in ds.mols]) == set([mol.xyz.shape for mol in ds2.mols]), f"shapes of xyz arrays are not the same: \n{set([mol.xyz.shape for mol in ds.mols])} \nvs \n{set([mol.xyz.shape for mol in ds2.mols])}"
#%%
ds.parametrize()
# %%
with tempfile.TemporaryDirectory() as tmpdirname:
    ds.save_npz(tmpdirname, overwrite=True)
    ds2 = PDBDataset.load_npz(tmpdirname)
glist = ds2.to_dgl()
assert "u_total_ref" in glist[0].nodes["g"].data.keys()
# %%
with tempfile.TemporaryDirectory() as tmpdirname:
    p = str(Path(tmpdirname)/Path("test_dgl.bin"))
    ds.save_dgl(p, overwrite=True)
    glist, _ = dgl.load_graphs(p)
    assert np.allclose(glist[0].nodes["g"].data["u_qm"], ds.to_dgl([0])[0].nodes["g"].data["u_qm"])

# %%
