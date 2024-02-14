"""
Assuming a dataset is stored as folder/molname/moldata.npz, this script creates a directory contianing molname.npz files.
"""

from pathlib import Path
from grappa.data import MolData
from grappa.utils import openmm_utils
import shutil
from typing import List
import numpy as np
from grappa.constants import CHARGE_MODELS

charge_model = 'classical'

def main(sourcepath, targetpath, openmm_ff:str=None, skip:List[str]=[]):
    if not openmm_ff is None:
        openmm_ff_name = openmm_ff.strip('.xml')
        openmm_ff_name = openmm_ff_name.strip('*')
        openmm_ff = openmm_utils.get_openmm_forcefield(openmm_ff)

    sourcepath = Path(sourcepath)
    targetpath = Path(targetpath)
    targetpath.mkdir(parents=True, exist_ok=True)
    for i,moldir in enumerate(sourcepath.iterdir()):
        if not moldir.is_dir():
            # copy file to target (can be README, etc.):
            shutil.copy(moldir, targetpath)
            continue

        print(f"Processing {i}, {moldir.name}        ", end='\r')
        try:
            molname = moldir.name
            moldata = MolData.load(moldir/'moldata.npz')
            assert charge_model in CHARGE_MODELS
            moldata.molecule.additional_features['charge_model'] = np.tile(np.array([cm == charge_model for cm in CHARGE_MODELS], dtype=np.float32), (len(moldata.molecule.atoms),1))
            if moldata.xyz.shape[0] == 0:
                print(f"Skipping {molname} because it has no conformers.")
                continue

            if moldata.pdb is not None and openmm_ff is not None:
                if molname in skip:
                    print(f"Skipping {molname}")
                    continue
                else:
                    moldata.calc_energies_openmm(openmm_forcefield=openmm_ff, forcefield_name=openmm_ff_name)
        except:
            print() # for not overwriting the previous line
            raise

        moldata.save(targetpath/molname)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, help='Path to the folder with npz files defining MolData objects.')
    parser.add_argument('--target_path', type=str, help='Path to the folder where the new MolData objects should be stored.')
    parser.add_argument('--openmm_ff', type=str, default=None, help='Path to the openmm forcefield xml file. If given, the energies and gradients are calculated using openmm. Can be eg "amber99sbildn.xml".')
    parser.add_argument('--skip', type=str, default=[], nargs='+', help='List of molnames to skip.')
    args = parser.parse_args()
    main(args.source_path, args.target_path, args.openmm_ff, skip=args.skip)