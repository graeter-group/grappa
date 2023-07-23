#%%
from grappa.PDBData.PDBDataset import PDBDataset

from grappa.constants import SPICEPATH
from pathlib import Path
dipeppath = str(Path(SPICEPATH).parent/Path("dipeptides_spice.hdf5"))

#%%
# ds = PDBDataset.from_spice(dipeppath, n_max=1, skip_errs=True)
# %%
ds2 = PDBDataset.from_spice(dipeppath, n_max=1, with_smiles=True)
# %%
ds2.parametrize("gaff-2.11")
# %%
mol = ds2[0]

# %%
mol.compare_with_ff(None, compare_ref=True)
# %%
