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

    # Update exclude list based on allow_collagen argument
    exclude = [] if args.allow_collagen else ["J", "O"]

    if args.folder is None:
        args.folder = f"pep_{args.length}"

    # Generate dataset
    if args.n_max > 0:
        assert args.length is not None, "Length must be specified when generating a fixed number of samples."
        outpath = str(Path(__file__).parent/args.folder)
        print(outpath)
        generate(n_max=args.n_max, length=args.length, outpath=outpath, exclude=exclude)

    for s in args.sequence:
        path = str(Path(__file__).parent/args.folder/s) if not args.folder is None else str(Path(__file__).parent/'data'/str(len(s)))
        
        os.makedirs(path, exist_ok=True)
        generate_peptide(code=s, dir=path, e=False, silent=True, overwrite=True, openmm_standard=True)