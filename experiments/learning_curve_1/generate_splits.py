from grappa.data import Dataset
import os
import json
from pathlib import Path

SPLITPATH = 'splits'

ds_tags = [
    "spice-dipeptide",
    "spice-des-monomers",
    "spice-pubchem",
    "gen2",
    "gen2-torsion",
    "pepconf-dlc",
    "protein-torsion",
    "rna-diverse",
]
# rna-nucleoside is pure train
# rna-trinucleotide is pure test


if __name__ == "__main__":

    ds = Dataset()
    for ds_tag in ds_tags:
        ds += Dataset.from_tag(ds_tag)

    for k in [3, 4, 6, 8]:

        splitpath = f"{SPLITPATH}_{k}"
        (Path(__file__).parent/splitpath).mkdir(exist_ok=True)

        split_dicts = ds.get_k_fold_split_ids(k=k, seed=42)

        for i, split_dict in enumerate(split_dicts):
            with open(Path(__file__).parent/splitpath/f"split_{i}.json", "w") as f:
                json.dump(split_dict, f, indent=4)