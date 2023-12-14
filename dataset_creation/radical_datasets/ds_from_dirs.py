"""
ASSUME THAT GRADIENTS ARE STORED WITH WRONG SIGN!
Assuming a dataset is stored as folder/molname/moldata.npz, this script creates a directory contianing molname.npz files.
"""

from pathlib import Path
from grappa.data import MolData

def main(sourcepath, targetpath):
    sourcepath = Path(sourcepath)
    targetpath = Path(targetpath)
    targetpath.mkdir(parents=True, exist_ok=True)
    for moldir in sourcepath.iterdir():
        if not moldir.is_dir():
            continue
        molname = moldir.name
        moldata = MolData.load(moldir/'moldata.npz')
        
        ###############################
        # reverse gradients:
        moldata.gradient = -moldata.gradient
        moldata.gradient_ref = -moldata.gradient_ref
        for v in moldata.ff_gradient.values():
            v *= -1
        for v in moldata.ff_nonbonded_gradient.values():
            v *= -1
        ###############################

        moldata.save(targetpath/molname)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, help='Path to the folder with npz files defining MolData objects.')
    parser.add_argument('--target_path', type=str, help='Path to the folder where the new MolData objects should be stored.')
    args = parser.parse_args()
    main(args.source_path, args.target_path)