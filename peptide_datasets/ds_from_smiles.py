"""
Parametrise the spice dipeptide dataset with the Amber 99SBildn forcefield and store the dataset as npz files representing a MolData object.
THIS DOES CURRENTLY NOT WORK BECAUSE THE MAPPED SMILES CONTAINS NO RESIDUE INFORMATION.
"""

raise NotImplementedError("This script does not work yet because the mapped smiles does not contain residue information.")

from grappa.data import MolData, Molecule
from pathlib import Path
import numpy as np
from openff.toolkit import Molecule as OFFMolecule
from openmm.app import ForceField

def main(source_path, target_path, forcefield):
    print(f"Converting\n{source_path}\nto\n{target_path}")
    source_path = Path(source_path)
    target_path = Path(target_path)

    target_path.mkdir(exist_ok=True, parents=True)

    # iterate over all child directories of source_path:
    num_total = 0
    num_success = 0
    num_err = 0

    total_mols = 0
    total_confs = 0

    for idx, molfile in enumerate(source_path.iterdir()):
        if molfile.is_dir():
            continue
        num_total += 1
        try:
            print(f"Processing {idx}", end='\r')
            data = np.load(molfile)
            # transform to actual dictionary
            data = {k:v for k,v in data.items()}

            xyz = data['xyz']
            energy = data['energy_qm']
            gradient = data['gradient_qm']

            mapped_smiles = data['mapped_smiles']
            if not isinstance(mapped_smiles, str):
                mapped_smiles = mapped_smiles[0]

            total_mols += 1
            total_confs += data['xyz'].shape[0]

            moldata = MolData.from_smiles(mapped_smiles=mapped_smiles, xyz=xyz, energy=energy, gradient=gradient, forcefield=forcefield, forcefield_type='openmm')

            moldata.save(target_path/(molfile.stem+'.npz'))

            num_success += 1
        except Exception as e:
            num_err += 1
            raise
            # print(f"Failed to process {molpath}: {e}")
            continue
    
    print("\nDone!")
    print(f"Processed {num_total} molecules, {num_success} successfully, {num_err} with errors")

    print(f"Total mols: {total_mols}, total confs: {total_confs}")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path",
        type=str,
        help="Path to the folder with npz files containing smiles, positions, energies and gradients.",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        help="Path to the target folder in which the dataset is stored as collection of npz files.",
    )
    parser.add_argument(
        "--forcefield",
        type=str,
        default='amber99sbildn.xml',
        help="Which forcefield to use for creating improper torsion and classical parameters. if no energy_ref and gradient_ref are given, the nonbonded parameters are used as reference.",
    )
    args = parser.parse_args()
    main(source_path=args.source_path, target_path=args.target_path, forcefield=args.forcefield)