"""
Writes smiles to the spice_dipeptide_amber99sbildn and tripeptides_amber99sbildn datasets if not present already and checks wheter the smiles inferred from the pdb files are consistent with those in the spice-dipeptide dataset.
Then, we can use the datasets together since the splitting is done consistently.
"""
#%%
from grappa.utils.dataset_utils import get_path_from_tag
from grappa.data import MolData
from pathlib import Path

from openff.toolkit.topology import Molecule as OFFMol
import tempfile

p = get_path_from_tag('spice_dipeptide_amber99sbildn') #, data_dir='test_data')

# add smiles to the dataset if not already present
def add_smiles(moldata):
    if moldata.smiles is not None:
        if moldata.mol_id != moldata.smiles:
            moldata.sequence = moldata.mol_id
            moldata.mol_id = moldata.smiles
        return moldata
    pdbstring = moldata.pdb
    with tempfile.TemporaryDirectory() as tmp:
        pdbpath = str(Path(tmp)/'pep.pdb')
        with open(pdbpath, "w") as pdb_file:
            pdb_file.write(pdbstring)
        openff_mol = OFFMol.from_polymer_pdb(pdbpath)

    this_smiles = openff_mol.to_smiles(mapped=False)

    moldata.smiles = this_smiles
    if moldata.mol_id != moldata.smiles:
        moldata.sequence = moldata.mol_id
        moldata.mol_id = moldata.smiles

    return moldata

#%%

smiles = []
for i, dipep_path in enumerate((Path(p).parents[1]/'grappa_datasets'/p.stem).iterdir()):
    print(i, end='\r')
    m = MolData.load(dipep_path)
    m = add_smiles(m)
    m.save(dipep_path)

    smiles.append(m.smiles)
# %%
ref_smiles=[]
for dipep_path in (Path(p).parents[1]/'grappa_datasets'/'spice-dipeptide').iterdir():
    mol = MolData.load(dipep_path).mol_id
    ref_smiles.append(MolData.load(dipep_path).mol_id)
# %%

# assert that all smiles can be found in the larger dataset
smiles = set(smiles)
ref_smiles = set(ref_smiles)

assert len(smiles) == len(ref_smiles.intersection(smiles))
# %%
for i, dipep_path in enumerate((Path(p).parents[1]/'grappa_datasets'/'tripeptides_amber99sbildn').iterdir()):
    print(i, end='\r')
    m = MolData.load(dipep_path)
    m = add_smiles(m)
    m.save(dipep_path)
# %%
