from __future__ import annotations

import argparse

from tqdm.auto import tqdm

from common import clamp_slice
from common import make_add_data
from common import make_mean_free_noise
from common import make_output_dir
from common import make_output_tag
from common import rng_for_slice
from common import sorted_npz_paths
from common import load_partial_charge_counts

from grappa.data import MolData
from grappa.utils import openmm_utils


SOURCE_TAG = "spice-dipeptide-amber99"
FORCEFIELD_XML = "amber99sbildn.xml"
FF_NAME = "amber99sbildn"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a noisy-charge amber99 dipeptide dataset.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for Gaussian charge noise.")
    parser.add_argument("--std", type=float, required=True, help="Standard deviation of the Gaussian charge noise.")
    parser.add_argument("--start", type=int, default=None, help="Inclusive start index in the numerically sorted source dataset.")
    parser.add_argument("--stop", type=int, default=None, help="Exclusive stop index in the numerically sorted source dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs that already exist.")
    args = parser.parse_args()

    source_paths = sorted_npz_paths(SOURCE_TAG)
    start, stop = clamp_slice(len(source_paths), args.start, args.stop)
    output_tag = make_output_tag(SOURCE_TAG, args.std, args.seed)
    output_dir = make_output_dir(SOURCE_TAG, args.std, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if start == stop:
        print(f"No work to do for {output_tag}: slice [{start}, {stop}) is empty.")
        return

    atom_counts = load_partial_charge_counts(source_paths)
    rng = rng_for_slice(args.seed, atom_counts, start)
    forcefield = openmm_utils.get_openmm_forcefield(FORCEFIELD_XML)

    print(f"Writing {output_tag} into {output_dir}")
    print(f"Processing indices [{start}, {stop}) out of {len(source_paths)} molecules")

    written = 0
    skipped = 0
    for path in tqdm(source_paths[start:stop], desc=output_tag):
        moldata = MolData.load(path)
        charges = moldata.molecule.partial_charges
        noise, noisy_charges = make_mean_free_noise(charges, rng, args.std)

        outpath = output_dir / path.name
        if outpath.exists() and not args.overwrite:
            skipped += 1
            continue

        topology = openmm_utils.topology_from_pdb(moldata.pdb)
        system = forcefield.createSystem(topology)
        system = openmm_utils.set_partial_charges(system, noisy_charges)

        new_moldata = MolData.from_openmm_system(
            openmm_system=system,
            openmm_topology=topology,
            xyz=moldata.xyz,
            gradient=moldata.gradient,
            energy=moldata.energy,
            mol_id=moldata.mol_id,
            pdb=moldata.pdb,
            sequence=moldata.sequence,
            mapped_smiles=moldata.mapped_smiles,
            smiles=moldata.smiles,
            ff_name=FF_NAME,
            allow_nan_params=True,
        )
        new_moldata.add_data = make_add_data(charges, noise, noisy_charges, args.std)
        new_moldata.save(outpath)
        written += 1

    print(f"Completed {output_tag}: wrote {written}, skipped {skipped}.")


if __name__ == "__main__":
    main()
