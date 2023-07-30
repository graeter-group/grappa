# %%
from grappa.constants import SPICEPATH
from grappa.PDBData.PDBDataset import PDBDataset
from pathlib import Path
#%%
# uncomment to create example:
spicepath = SPICEPATH
dipeppath = str(Path(spicepath).parent/Path("dipeptides_spice.hdf5"))
print(dipeppath)
ds = PDBDataset.from_spice(dipeppath, n_max=1, randomize=True, skip_errs=True)
m = ds[0]
storepath = Path(__file__).parent.parent.parent.parent/Path("src/grappa/PDBData")
m.save(storepath/Path("example_PDBMolecule.npz"))
# %%
from grappa.PDBData.PDBMolecule import PDBMolecule
mol = PDBMolecule.get_example()
mol.gradients.shape
# %%
# from grappa.ff import ForceField
# ff = ForceField.from_tag("example")

# fig, ax = mol.compare_with_ff(ff, ff_title="grappa")

# %%
import openmm.app 
fig, ax = mol.compare_with_ff(openmm.app.ForceField("amber99sbildn.xml"), ff_title="amber")
# %%
