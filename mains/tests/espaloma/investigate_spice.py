#%%
from grappa.PDBData.PDBMolecule import PDBMolecule
from grappa.PDBData.PDBDataset import PDBDataset

from grappa.constants import DEFAULTBASEPATH
from pathlib import Path

storagepath = str(Path(DEFAULTBASEPATH) / "qca_spice/charge_default_ff_gaff-2_11_filtered_with_esp")


spice_espaloma = PDBDataset.load_npz(storagepath, n_max=2)


# %%
spice_espaloma.evaluate(plotpath="dipeptides/plots", suffix="_total_ref", plot_args={"ff_title": "Espaloma"})
# %%
spice_espaloma[0].graph_data["g"]["u_esp"]
# %%
mol = spice_espaloma[0]
mol.graph_data["n2"]["eq_"]