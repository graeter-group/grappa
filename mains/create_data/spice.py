#%%
from grappa.PDBData.PDBDataset import PDBDataset
from pathlib import Path
import shutil
import os
from grappa.constants import SPICEPATH as spicepath
dipeppath = str(Path(spicepath).parent/Path("dipeptides_spice.hdf5"))

storepath = Path(spicepath).parent/Path("PDBDatasets/spice/base")

if __name__=="__main__":

    ds = PDBDataset.from_spice(dipeppath, info=True, n_max=None)

    # remove conformations with energy > 200 kcal/mol from min energy in ds[i]
    ds.filter_confs(max_energy=200, max_force=500) # remove crashed conformations

    # delete folder is present:
    if os.path.exists(str(storepath)):
        shutil.rmtree(str(storepath))

    ds.save_npz(storepath, overwrite=True)
    ds.save_dgl(str(storepath)+"_dgl.bin", overwrite=True)