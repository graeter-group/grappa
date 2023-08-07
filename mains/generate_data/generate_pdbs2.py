import argparse
from pepgen.dataset import generate
from pepgen.pepgen import generate_peptide
from pathlib import Path
import os

if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser(description='Script to generate dataset using Pepgen.')
    parser.add_argument('--n_max', '-n', type=int, help='Maximum number of samples to generate. If possible number of sequences is smaller, only these are generated.', default=0)
    parser.add_argument('--length', '-l', type=int, help='Length of the peptide sequences.', default=None)
    parser.add_argument('--allow_collagen', action='store_true', help='Allow collagen in the dataset.')
    parser.add_argument('--folder', type=str, help='Output folder for the data.', default=None)
    parser.add_argument('--sequence', '-s', type=str, nargs='+', help='Single letter sequence.', default=[])
    
    args = parser.parse_args()

    print(f"Generating dataset with {args.sequence}.")

    for s in args.sequence:
        
        os.makedirs(args.folder, exist_ok=True)
        generate_peptide(code=s, dir=args.folder, e=False, silent=True, overwrite=True, openmm_standard=True)