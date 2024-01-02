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
    (Path(__file__).parent/SPLITPATH).mkdir(exist_ok=True)

    ds = Dataset()
    for ds_tag in ds_tags:
        ds += Dataset.from_tag(ds_tag)

    split_dicts = ds.get_k_fold_split_ids(k=10, seed=42)

    for i, split_dict in enumerate(split_dicts):
        with open(Path(__file__).parent/SPLITPATH/f"split_{i}.json", "w") as f:
            json.dump(split_dict, f, indent=4)