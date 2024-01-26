
from grappa.data import MolData
from pathlib import Path
import numpy as np
import traceback

def main(source_path, target_path, forcefield='openff_unconstrained-2.0.0.offxml', partial_charge_key='am1bcc_elf_charges'):
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

    num_nan_params = 0

    for idx, molfile in enumerate(source_path.iterdir()):
        if molfile.is_dir():
            continue
        num_total += 1
        try:
            print(f"Processing {idx}", end='\r')
            data = np.load(molfile)
            # ransform to actual dictionary
            data = {k:v for k,v in data.items()}
            try:
                moldata = MolData.from_data_dict(data_dict=data, partial_charge_key=partial_charge_key, forcefield=forcefield)
            except:
                moldata = MolData.from_data_dict(data_dict=data, partial_charge_key=partial_charge_key, forcefield=forcefield, allow_nan_params=True)
                num_nan_params += 1

            # moldata.molecule.add_features(['ring_encoding'])

            total_mols += 1
            total_confs += data['xyz'].shape[0]

            moldata.save(target_path/(molfile.stem+'.npz'))

            num_success += 1
        except Exception as e:
            num_err += 1
            # get traceback:
            tb = traceback.format_exc()
            print(f"\nError processing {molfile} with smiles {data['smiles']}: {e}\n{tb}\n")
            continue
    
    print("\nDone!")
    print(f"Processed {num_total} molecules, {num_success} successfully, {num_err} with errors, {num_nan_params} with nan params.")

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
        default='openff_unconstrained-2.0.0.offxml',
        help="Which forcefield to use for creating improper torsion and classical parameters. if no energy_ref and gradient_ref are given, the nonbonded parameters are used as reference.",
    )
    parser.add_argument(
        "--partial_charge_key",
        type=str,
        default='am1bcc_elf_charges',
        help="Which partial charge key to use for creating improper torsion and classical parameters. if no energy_ref and gradient_ref are given, the nonbonded parameters are used as reference.",
    )
    args = parser.parse_args()
    if args.partial_charge_key in ['None', 'none', '']:
        args.partial_charge_key = None
        
    main(source_path=args.source_path, target_path=args.target_path, forcefield=args.forcefield, partial_charge_key=args.partial_charge_key)