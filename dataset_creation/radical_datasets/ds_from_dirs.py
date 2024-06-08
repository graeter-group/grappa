"""
Assuming a dataset is stored as folder/molname/moldata.npz, this script creates a directory containing molname.npz files.
"""

from pathlib import Path
from grappa.data import MolData, Molecule
from grappa.utils import openmm_utils
import shutil
from typing import List
import numpy as np
from grappa.constants import CHARGE_MODELS
import numpy as np

charge_model = 'amber99'

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

        try:
            molname = moldir.name
            data = {k:v for k,v in np.load(moldir/'moldata.npz').items()}

            energy = data.pop('energy_qm')
            gradient = data.pop('gradient_qm')
            xyz = data.pop('xyz')
            pdb = data.pop('pdb')
            sequence = str(data.pop('sequence'))

            oneletter_code = {
                "ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "CYS":"C", "GLN":"Q", "GLU":"E", "GLY":"G", "HIS":"H", "ILE":"I", "LEU":"L", "LYS":"K", "MET":"M", "PHE":"F", "PRO":"P", "SER":"S", "THR":"T", "TRP":"W", "TYR":"Y", "VAL":"V",
                "HYP": "O", "DOP": "J",
                "ACE": "B", "NME": "Z",
                "HID": "1", "HIP": "2", "HIE": "H",
                "ASH": "3", "GLH": "4",
                "LYN": "5", "CYX": "6",
            }
            
            # convert sequence:
            # strip ACE and NME:
            sequence = sequence.replace('ACE-', '').replace('-NME', '')
            threeletter_AAs = sequence.split('-')
            sequence = ''.join([oneletter_code[aa] for aa in threeletter_AAs])
            sequence += '_radical'

            print(f"Processing {i}, {sequence}        ")

            mol_id = data.pop('mol_id')
            energy_ref = data.pop('energy_ref')
            gradient_ref = data.pop('gradient_ref')

            molecule = Molecule(atoms=data['atoms'], bonds=data['bonds'], impropers=data['impropers'], atomic_numbers=data['atomic_numbers'], partial_charges=data['partial_charges'])
            molecule.charge_model = "amber99"

            moldata = MolData(molecule=molecule, xyz=xyz, energy=energy, gradient=gradient, mol_id=mol_id, sequence=sequence, pdb=pdb, ff_energy={'reference_ff':{'nonbonded':energy - energy_ref}}, ff_gradient={'reference_ff':{'nonbonded':gradient - gradient_ref}})

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