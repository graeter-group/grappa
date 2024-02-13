from grappa.data import MolData, Dataset
from pathlib import Path
import numpy as np

def main(source_path, target_path):
    print(f"Converting\n{source_path}\nto\n{target_path}")
    source_path = Path(source_path)
    target_path = Path(target_path)

    target_path.mkdir(exist_ok=True, parents=True)

    total_mols = 0
    total_confs = 0

    mols = []

    for idx, molfile in enumerate(source_path.iterdir()):
        if molfile.is_dir():
            continue

        if not molfile.name.endswith('.npz'):
            continue

        if molfile.name.endswith('.md') or molfile.name.endswith('.txt'):
            # assume that there is some information written into a text file, copy the file:
            target_file = target_path / molfile.name
            molfile.copy(target_file)

        print(f"Processing {idx}", end='\r')

        moldata = MolData.load(str(molfile))

        moldata.molecule.add_features(['ring_encoding', 'degree'])

        total_mols += 1
        total_confs += len(moldata.molecule.atoms)

        mols.append(moldata)

    assert not len(mols) == 0, "No molecules found!"

    ds = Dataset.from_moldata(mols, subdataset=source_path.stem)
    
    assert len(ds) == total_mols

    ds.save(target_path)

    print("\nDone!")

    print(f"Total mols: {len(ds)}, total confs: {total_confs}")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path",
        type=str,
        help="Path to the folder with npz files defining MolData objects.",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        help="Path to the target folder in which the dataset is stored as graphs.bin file and json lists.",
    )
    args = parser.parse_args()
    main(source_path=args.source_path, target_path=args.target_path)
