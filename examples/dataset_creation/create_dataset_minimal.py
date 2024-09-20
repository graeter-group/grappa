#%%
"""
MINIMAL EXAMPLE FOR CREATING A DATASET FROM PDB FILES AND QM DATA.

Grappa is trained on bonded energies and forces, that is, energies and forces from QM calculations subtracted by the nonbonded contribution assigned by a classical force field of choice.

Its input to a grappa model is the molecular graph equipped with partial charges used for calculating the nonbonded contribution, thus the dataset must contain for every molecule:
- Atomic numbers
- Partial charges
- Connectivity
- (Optionally: Bonded parameters of the classical force field of choice for accelerated pretraining)

The conformational training data is
- Atomic positions
- Bonded energy contribution (QM minus nonbonded of classical force field)
- Bonded force contribution (QM minus nonbonded of classical force field)

Additionally, a molecular identifier (a string) is required to make the split into train/val/test set reproducible and reliable in the case of having several conformations of the same molecule in different subdatasets.

In this example, we use a small dataset of QM energies and force for peptides and functionality from the grappa package to calculate the nonbonded contribution of the Amber99SBILDN force field in OpenMM. Thus, the OpenMM-Grappa installation is needed.
"""

import numpy as np
from pathlib import Path

from grappa.data import MolData
from openmm.app import PDBFile, ForceField
from tqdm import tqdm

#%%

# Load the dataset
thisdir = Path(__file__).parent
dspath = thisdir / "minimal_example_data"

data = []

for i in range(4):
    path = dspath / f'example_data_{i}.npz'
    moldata = np.load(path)
    data.append({
        'xyz': moldata['xyz'], # shape (n_confs, n_atoms, 3)
        'energy_qm': moldata['energy_qm'], # shape (n_confs,)
        'gradient_qm': moldata['gradient_qm'], # negative force, shape (n_confs, n_atoms, 3)
    })
    
# %%

# now we can create an openmm system and topology, which we pass to a constructur of Grappa's MolData class that handles the calculation of the nonbonded contribution


for i in tqdm(range(len(data))):
    xyz = data[i]['xyz']
    energy_qm = data[i]['energy_qm']
    gradient_qm = data[i]['gradient_qm']
    identifier = f"example_{i}" # this should ideally be something general and unique, i.e. a unique smiles string or a sequence

    topology = PDBFile(str(dspath/f"pdb_{i}.pdb")).topology
    system = ForceField('amber99sbildn.xml').createSystem(topology)

    moldata = MolData.from_openmm_system(openmm_topology=topology, openmm_system=system, xyz=xyz, energy=energy_qm, gradient=gradient_qm, mol_id=identifier)

    # while not necessary for training the model, it is usually a good idea to also store the pdbfile in the MolData object to enable other users the reconstruction of the system, which will be much more difficult if residue information is lost. (simply set pdb=pdbstring as kwarg in the MolData.from_openmm_system call)

    moldata.save(dspath/f"moldata_{i}.npz")

#%%
"""
If we want to train a grappa model, we need to store the dataset as dgl graphs and point the model towards the dataset.
The easiest way to do so is to store the dataset at grappa.utils.get_data_path()/grappa_datasets/<some_name>/*.npz and give the model <some_name> as dataset tag. Then, it can be loaded by grappa.data.Dataset.from_tag('<some_name>').
"""

from grappa.utils.data_utils import get_data_path

# load the moldata objects
moldata_paths = list(dspath.glob("moldata_*.npz"))
data = [MolData.load(p) for p in moldata_paths]

# define the dataset name and the canonical path to store the dataset
dsname = "minimal_example_dataset"
dataset_folder = get_data_path()/'datasets'/dsname

# store the moldata objects as npz files there:
for i, moldata in enumerate(data):
    moldata.save(dataset_folder/f"{i}.npz")

#%%
# now we can load a processed torch dataset from the tag:
from grappa.data import Dataset, clear_tag

# if you change data, make sure to delete the old preprocessed dataset (which is stored in a cache-dataset-folder)
clear_tag(dsname)

dataset = Dataset.from_tag(dsname)
print("length of the dataset:", len(dataset))

#%%

s = """
Now we can train and evaluate on these datasets by appending these tags to a config file as done in configs/minimal_example.yaml.
To train on these datasets we just created, run 'python experiments/train.py data=minimal_example' from the root directory.
Stop the training run by pressing ctrl+c.
Find the checkpoint path at 'ckpt/grappa/baseline/<date>/...'.
Evaluate it by running 'python experiments/evaluate.py ckpt_path=<path_to_checkpoint>'.
"""
print(s)
# %%

"""
In case you want to create a dataset to fine tune grappa for a single, specific molecule or few molecules and don't want to split the dataset according to the molecule identifier but just according to different conformations, it is best practice to split the dataset yourself and store three datasets for train, val and test set and store them separately with different tags. Then set
config.data.pure_train_datasets += [train_tag]
config.data.pure_val_datasets += [val_tag]
config.data.pure_test_datasets += [test_tag]
"""
# %%
