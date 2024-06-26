#%%
"""
Grappa is trained on bonded energies and forces, that is, energies and forces from QM calculations subtracted by the nonbonded contribution assigned by a classical force field of choice.

Its input to a grappa model is the molecular graph equipped with partial charges used for calculating the nonbonded contribution, thus the dataset must contain for every molecule:
- Atomic numbers
- Partial charges
- Connectivity
- (Optionally: Bonded parameters of the classical force field of choice for accelerated pretraining)

The conformational training data itself is
- Atomic positions
- Bonded energy contribution (QM minus nonbonded of classical force field)
- Bonded force contribution (QM minus nonbonded of classical force field)

Additionally, a molecular identifier (a string) is required to make the split into train/val/test set reproducible and reliable in the case of having several conformations of the same molecule in different subdatasets.

Grappa can also be trained to be consistent with nonbonded contributions from several classical force fields. In this case, the data should be annotated by a flag describing which classical force field was used to calculate the repective nonbonded contribution, given by the charge_model keyword in the Grappa's MolData class.

In this example, we use a small dataset of QM energies and force for peptides and functionality from the grappa package to calculate the nonbonded contribution of the Amber99SBILDN force field in OpenMM. Thus, the OpenMM-Grappa installation is needed.
"""

import numpy as np
from pathlib import Path

from grappa.data import MolData
from grappa.utils.openmm_utils import get_openmm_forcefield

import logging
# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#%%

# Load the dataset
thisdir = Path(__file__).parent
dspath = thisdir / "tripeptide_example_data"

data = []

for npzfile in dspath.glob("example_data_*.npz"):
    moldata = np.load(npzfile)
    data.append({
        'xyz': moldata['xyz'],
        'energy_qm': moldata['energy_qm'],
        'gradient_qm': moldata['gradient_qm'], # negative force
        'sequence': moldata['sequence'], # used as identifier
        'pdbstring': moldata['pdbstring'], # required to apply classical force field and used to obtain atomic numbers and connectivity
    })


# %%
# Investigate this example data:
print(f"xyz.shape = (n_confs, n_atoms, 3): {data[0]['xyz'].shape}")
print(f"energy_qm.shape = (n_confs,): {data[0]['energy_qm'].shape}")
print(f"gradient_qm.shape = (n_confs, n_atoms, 3): {data[0]['gradient_qm'].shape}")
print(f"sequence: {data[0]['sequence']}")
# %%
print("The pdbstring is a list of lines of the pdbfile\nof some conformation of the respective peptide:\n")
print(data[0]['pdbstring'][0])
print(data[0]['pdbstring'][1])
print(data[0]['pdbstring'][2])
print(data[0]['pdbstring'][10])

print("We can save the pdbstring as file to create an openmm system later on:")
for i, mol in enumerate(data):
    with open(dspath/f"pdb_{i}.pdb", 'w') as f:
        f.write(''.join(mol['pdbstring']))
# %%

# now we can create an openmm system and topology using the charmm36 force field and the pdbfile, which we pass to a constructur of Grappa's MolData class that handles the calculation of the nonbonded contribution

from openmm.app import PDBFile, ForceField
from tqdm import tqdm

for i in tqdm(range(len(data))):
    xyz = data[i]['xyz']
    energy_qm = data[i]['energy_qm']
    gradient_qm = data[i]['gradient_qm']
    sequence = data[i]['sequence']

    topology = PDBFile(str(dspath/f"pdb_{i}.pdb")).topology
    # system = ForceField('amber99sbildn.xml').createSystem(topology)
    # for charmm, we need a modified version that can deal with ACE and NME caps:
    system = get_openmm_forcefield('charmm36').createSystem(topology)

    moldata = MolData.from_openmm_system(openmm_topology=topology, openmm_system=system, xyz=xyz, energy=energy_qm, gradient=gradient_qm, mol_id=sequence, charge_model='charmm')

    # while not necessary for training the model, it is usually a good idea to also store the pdbfile in the MolData object to enable other users the reconstruction of the system, which will be much more difficult if residue information is lost. (simply set pdb=pdbstring as kwarg in the MolData.from_openmm_system call)

    moldata.save(dspath/f"moldata_{i}.npz")

#%%
# it is also possible to pass the partial charges manually, in which case they overwrite those in the openmm system before calculating the nonbonded contribution:
i = 0
xyz, energy_qm, gradient_qm, sequence = data[i]['xyz'], data[i]['energy_qm'], data[i]['gradient_qm'], data[i]['sequence']

topology = PDBFile(str(dspath/f"pdb_{i}.pdb")).topology
system = get_openmm_forcefield('charmm36').createSystem(topology)

# use random partial charges:
partial_charges = np.random.rand(topology.getNumAtoms())
partial_charges -= partial_charges.mean()

moldata_own_charges = MolData.from_openmm_system(openmm_topology=topology, openmm_system=system, xyz=xyz, energy=energy_qm, gradient=gradient_qm, mol_id=sequence, charge_model='charmm', partial_charges=partial_charges)

# if the charges are changed systematically, one could modify grappa.constants.CHARGE_MODELS to include a new charge model, e.g. 'amber99_modified'/'amber99_overcharged'/... and pass this as charge model.

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
dsname = "example_dataset"
dataset_folder = get_data_path()/'datasets'/dsname

# store the moldata objects as npz files there:
for i, moldata in enumerate(data):
    moldata.save(dataset_folder/f"{i}.npz")

#%%
# now we can load a processed torch dataset from the tag:
from grappa.data import Dataset, clear_tag

# if you change data, make sure to delete the old processed dataset:
clear_tag(dsname)

dataset = Dataset.from_tag(dsname)
print("length of the dataset:", len(dataset))
# %%

# we can also train a model on this dataset by putting its name in the train config file
from grappa.training import Experiment
from grappa.utils import get_repo_dir
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
import os
import torch

config_dir = get_repo_dir() / "configs"

# Compute the relative path from the current directory to the config directory (hydra needs it to be relative)
relative_path = os.path.relpath(config_dir, Path.cwd())

#%%
initialize(config_path=relative_path)
#%%

# Get the default config for training
config = compose(config_name="train")

# set the datasets
config.data.datasets = [dsname, 'spice-dipeptide']
config.data.pure_train_datasets = []
config.data.pure_val_datasets = []
config.data.pure_test_datasets = []

config.experiment.trainer.max_epochs = 50 if torch.cuda.is_available() else 5
config.experiment.trainer.accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

experiment = Experiment(config)
experiment.train()
#%%
experiment.test()

#%%
"""
In case you want to create a dataset to fine tune grappa for a single, specific molecule or few molecules and don't want to split the dataset according to the molecule identifier but just according to different conformations, it is best practice to split the dataset yourself and store three datasets for train, val and test set and store them separately with different tags. Then set
config.data.pure_train_datasets += [train_tag]
config.data.pure_val_datasets += [val_tag]
config.data.pure_test_datasets += [test_tag]
"""
# %%
