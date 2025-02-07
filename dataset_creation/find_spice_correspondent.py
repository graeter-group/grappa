from grappa.data import Dataset, MolData
from grappa.utils import get_data_path
from grappa.utils.graph_utils import get_isomorphisms
from tqdm import tqdm
from pathlib import Path

def redefine_mol_id(dspath:Path):
    """
    Define the mol_ids of the dataset dspath/*.npz as smilestring of the corresponding isomorphic spice entry.
    """
    assert dspath.is_dir()
    mol_data_paths = list(dspath.glob("*.npz"))
    assert len(mol_data_paths) > 0

    spice_dipeptide = Dataset.from_tag("spice-dipeptide")

    this_ds = [MolData.load(path).to_dgl() for path in tqdm(list(mol_data_paths), desc="Loading graphs")]

    isomorphisms = get_isomorphisms(this_ds, spice_dipeptide.graphs)

    assert len(isomorphisms) == len(mol_data_paths)

    for this_idx, spice_idx in tqdm(isomorphisms, desc="Assigning mol_ids"):
        mol_data = MolData.load(mol_data_paths[this_idx])
        mol_data.mol_id = spice_dipeptide.mol_ids[spice_idx]
        mol_data.save(mol_data_paths[this_idx])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dspath", type=str, help="Path to the peptide dataset where the mol id should be re-defined such that it agrees with the spice-dipeptide dataset. Can only be used for dipeptide datasets!")
    args = parser.parse_args()

    # if dspath is a string, not a path:
    if not "/" in args.dspath:
        args.dspath = get_data_path()/"datasets"/args.dspath

    redefine_mol_id(Path(args.dspath))