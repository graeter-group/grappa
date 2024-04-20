"""
Parametrise the spice dipeptide dataset with the Amber 99SBildn forcefield and store the dataset as npz files representing a MolData object.
"""


from grappa.data import MolData, Molecule
from pathlib import Path
import numpy as np
from openmm.app import ForceField
import tempfile
from openmm.app import PDBFile
from grappa.utils import openmm_utils, openff_utils

def main(source_path, target_path):
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
            #print(f"Processing {idx}", end='\r')
            data = np.load(molfile)
            # transform to actual dictionary
            data = {k:v for k,v in data.items()}

            # go from (n_atoms, n_confs, 3) to (n_confs, n_atoms, 3)
            xyz = data['n1 xyz'].transpose(1,0,2)
            gradient = data['n1 grad_qm'].transpose(1,0,2)
            energy = data['g u_qm'][0]
            pdb = data['pdb'].tolist()
            pdbstring = ''.join(pdb)
            sequence = str(data['sequence'])

            energy_nonbonded = data['g u_nonbonded_ref'][0]
            energy_total_ff = data['g u_total_ref'][0]

            gradient_nonbonded = data['n1 grad_nonbonded_ref'].transpose(1,0,2)
            gradient_total_ff = data['n1 grad_total_ref'].transpose(1,0,2)

            atoms = list(range(xyz.shape[1]))
            bonds = data['n2 idxs'].tolist()
            impropers = data['n4 idxs'].tolist()

            atomic_numbers = data['n1 atomic_number']
            if len(atomic_numbers.shape) == 1:
                atomic_numbers = atomic_numbers.tolist()
            elif len(atomic_numbers.shape) == 2:
                atomic_numbers = np.argmax(atomic_numbers, axis=1).tolist()
            else:
                raise ValueError(f"Atomic numbers have shape {atomic_numbers.shape}, should be 1 or 2 dimensional.")
            assert min(atomic_numbers) > 0, f"Atomic numbers should be > 0, but are {atomic_numbers}"

            is_radical = data['n1 is_radical']
            is_radical = np.array(is_radical).reshape(-1)
            partial_charge = data['n1 q_ref'].tolist()

            mol = Molecule(atoms=atoms, bonds=bonds, impropers=impropers, atomic_numbers=atomic_numbers, partial_charges=partial_charge, charge_model='amber99')

            mol.additional_features.update({'is_radical': is_radical})

            total_mols += 1
            total_confs += len(energy)

            print(f"Processing {idx}, sequence: {sequence}")


            # create moldata object without amber99 parameters
            moldata = MolData.from_arrays(molecule=mol, xyz=xyz, gradient=gradient, energy=energy, sequence=sequence, nonbonded_energy=energy_nonbonded, nonbonded_gradient=gradient_nonbonded, ff_energy=energy_total_ff-energy_nonbonded, ff_gradient=gradient_total_ff-gradient_nonbonded)

            moldata.molecule.add_features(['ring_encoding'])

            moldata.pdb = pdbstring

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

    args = parser.parse_args()
    main(source_path=args.source_path, target_path=args.target_path)