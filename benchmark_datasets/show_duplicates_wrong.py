"""
The duplicates stated from espaloma are incomplete! This is a demonstration of that.
"""

#%%
from pathlib import Path
import numpy as np


def check_if_duplicate(mol1, mol2):
    p = '/hits/fast/mbm/seutelf/data/datasets'

    data1 = np.load(Path(p)/mol1[0]/(mol1[1]+'.npz'))
    data2 = np.load(Path(p)/mol2[0]/(mol2[1]+'.npz'))

    smiles1 = data1['smiles'].item()
    smiles2 = data2['smiles'].item()

    print('smiles are the same:', smiles1 == smiles2)
    # now show that these are not included in duplicates

    duplicates = Path('/hits/fast/mbm/seutelf/esp_data/duplicated-isomeric-smiles-merge')

    found = False

    for subdir in duplicates.iterdir():
        for subsubdir in subdir.iterdir():
            for mol in [mol1, mol2]:
                if subsubdir.stem == mol[0]:
                    for molpath in subsubdir.iterdir():
                        if molpath.stem == mol[1]:
                            print((subsubdir.stem, molpath.stem), 'is in duplicates')
                            found = True
    if not found:
        print('not found!')

#%%
mol1 = ('rna-diverse', '62')
mol2 = ('rna-trinucleotide', '39')
check_if_duplicate(mol1, mol2)

# %%
# example for two that are actually in duplicates:
mol1 = ('gen2', '861')
mol2 = ('gen2-torsion', '744')
check_if_duplicate(mol1, mol2)
# %%
