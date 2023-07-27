#%%
from grappa.PDBData.PDBMolecule import PDBMolecule
from grappa.PDBData.PDBDataset import PDBDataset

from grappa.constants import DEFAULTBASEPATH
from pathlib import Path

storagepath = str(Path(DEFAULTBASEPATH) / "monomers/charge_default_ff_gaff-2_11_filtered_with_esp")


spice_espaloma = PDBDataset.load_npz(storagepath, n_max=2)


# %%
spice_espaloma.evaluate(plotpath="monomers/plots", suffix="_esp", plot_args={"ff_title": "Espaloma"})
# %%
spice_espaloma[0].graph_data["g"]["u_esp"]
# %%
