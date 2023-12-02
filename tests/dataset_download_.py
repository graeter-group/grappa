#%%
from grappa.utils.dataset_utils import get_path_from_tag

#%%
p = get_path_from_tag('tripeptides_amber99sbildn') #, data_dir='test_data')
#%%
from grappa.data import MolData
from pathlib import Path

# %%
from openff.toolkit.topology import Molecule as OFFMol
import tempfile

p = get_path_from_tag('spice_dipeptide_amber99sbildn') #, data_dir='test_data')


smiles = []
for i, dipep_path in enumerate((Path(p).parents[1]/'grappa_datasets'/p.stem).iterdir()):
    print(i, end='\r')
    m = MolData.load(dipep_path)
    pdbstring = m.pdb
    with tempfile.TemporaryDirectory() as tmp:
        pdbpath = str(Path(tmp)/'pep.pdb')
        with open(pdbpath, "w") as pdb_file:
            pdb_file.write(pdbstring)
        openff_mol = OFFMol.from_polymer_pdb(pdbpath)

    smiles.append(openff_mol.to_smiles())
# %%
ref_smiles=[]
for dipep_path in (Path(p).parents[1]/'grappa_datasets'/'spice-dipeptide').iterdir():
    ref_smiles.append(MolData.load(dipep_path).mol_id)
# %%
smiles = set(smiles)
ref_smiles = set(ref_smiles)

assert len(smiles) == len(ref_smiles.intersection(smiles))