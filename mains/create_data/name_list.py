"""
Python script that accepts the path to a PDBDataset and prints the list of names that it contains.
"""
#%%
from grappa.PDBData.PDBDataset import PDBDataset
from pathlib import Path
import argparse
from grappa.constants import DS_PATHS

#%%

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to PDBDataset. May also be a tag for grappa.constants.DS_PATHS.")
args = parser.parse_args()

path = args.path

if path in DS_PATHS.keys():
    path = DS_PATHS[path]


#%%
# path = DS_PATHS["spice_pubchem"]
ds = PDBDataset.load_npz(path, info=False)
names = [mol.name for mol in ds]
print(*names, sep=" ")
# %%
