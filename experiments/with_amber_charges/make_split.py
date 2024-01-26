"""
Creates a split of the tripeptide dataset and adds it to the espaloma split.
"""

#%%
from grappa.data import Dataset
import json
from pathlib import Path

tripep = Dataset.from_tag("tripeptides_amber99sbildn")

# ignore the fact that there are some capped peptides in the protein torsion dataset already since we have no smiles as of now (not so hard to change this actually...)
tripep += Dataset.from_tag("capped_peptide_amber99sbildn")

tripep_split = tripep.calc_split_ids((0.8,0.1,0.1), seed=42)

with open('tripep_split.json', 'w') as fp:
    json.dump(tripep_split, fp)

esp_split_path = str(Path(__file__).parent.parent.parent/f"dataset_creation/get_espaloma_split/espaloma_split.json")

with open(esp_split_path, "r") as f:
    esp_split = json.load(f)

for k in esp_split.keys():
    esp_split[k] = esp_split[k] + tripep_split[k]

with open('split.json', 'w') as fp:
    json.dump(esp_split, fp)
# %%
dipep = Dataset.from_tag("dipeptides_amber99sbildn")

for id in dipep.mol_ids:
    if not any([id in split for split in esp_split.values()]):
        print(id)
# %%
