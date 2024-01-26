"""
NOTE: CAN BE DELETED IN THE FUTURE, NOT NEEDED ANYMORE.
"""

from pathlib import Path
from grappa.data import MolData

DATAPATH = Path(__file__).parent.parent/"data/grappa_datasets"

for dspath in DATAPATH.iterdir():
    if dspath.is_dir():
        print(f"Processing {dspath.name}...")
        for i, npzpath in enumerate(dspath.iterdir()):
            if npzpath.suffix == ".npz":
                print(f"Processing {i}", end="\r")
                mol = MolData.load(str(npzpath))
                mol.molecule.add_features("mass")
                mol.save(str(npzpath))