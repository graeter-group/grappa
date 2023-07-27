#%%
from grappa.PDBData.PDBMolecule import PDBMolecule
from grappa.PDBData.PDBDataset import PDBDataset

from grappa.constants import DEFAULTBASEPATH

from pathlib import Path

monomers = PDBDataset.load_npz(str(Path(DEFAULTBASEPATH) / "monomers/charge_default_ff_gaff-2_11_filtered"), n_max=10)

mol = monomers[0]
# %%
mol.compare_with_espaloma()
mol.graph_data["g"]["u_total_ref"] = mol.graph_data["g"]["u_esp"]
mol.graph_data["n1"]["grad_total_ref"] = mol.graph_data["n1"]["grad_esp"]

fig, ax = mol.compare_with_ff()
# save the figure:
fig.savefig("monomer_compare.png", dpi=300)
# %%
print(mol.graph_data["g"]["u_esp"])

mol.save("monomer_with_esp.npz")

print(mol.graph_data["g"]["u_pred_esp"])