#%%
from grappa.PDBData.PDBMolecule import PDBMolecule
from grappa.PDBData.PDBDataset import PDBDataset

from grappa.constants import DEFAULTBASEPATH

from pathlib import Path

spice = PDBDataset.load_npz(str(Path(DEFAULTBASEPATH) / "monomers/charge_default_ff_gaff-2_11_filtered"))

for i in range(len(spice)):
    spice[i].compare_with_espaloma()

spice.save_npz(str(Path(DEFAULTBASEPATH) / "monomers/charge_default_ff_gaff-2_11_filtered_with_esp"), overwrite=True)

# %%
