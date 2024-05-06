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

Grappa can also be trained to be consistent with nonbonded contributions from several classical force fields. In this case, the data should be annotated by a flag describing which classical force field was used to calculate the repective nonbonded contribution.

In this example, we use a small dataset of QM energies and force for peptides and functionality from the grappa package to calculate the nonbonded contribution of the Amber99SBILDN force field in OpenMM. Thus, the OpenMM-Grappa installation is needed.
"""

import numpy as np
from pathlib import Path

from grappa.data import MolData

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
    system = ForceField('amber99sbildn.xml').createSystem(topology)

    moldata = MolData.from_openmm_system(openmm_topology=topology, openmm_system=system, xyz=xyz, energy=energy_qm, gradient=gradient_qm, mol_id=sequence, charge_model='amber99')

    # while not necessary for training the model, it is usually a good idea to also store the pdbfile in the MolData object to enable other users the reconstruction of the system, which will be much more difficult if residue information is lost. (simply set pdb=pdbstring as kwarg in the MolData.from_openmm_system call)

    moldata.save(dspath/f"moldata_{i}.npz")

#%%
# it is also possible to pass the partial charges manually, in which case they overwrite those in the openmm system before calculating the nonbonded contribution:
i = 0
xyz, energy_qm, gradient_qm, sequence = data[i]['xyz'], data[i]['energy_qm'], data[i]['gradient_qm'], data[i]['sequence']

topology = PDBFile(str(dspath/f"pdb_{i}.pdb")).topology
system = ForceField('amber99sbildn.xml').createSystem(topology)

# use random partial charges:
partial_charges = np.random.rand(topology.getNumAtoms())
partial_charges -= partial_charges.mean()

moldata_own_charges = MolData.from_openmm_system(openmm_topology=topology, openmm_system=system, xyz=xyz, energy=energy_qm, gradient=gradient_qm, mol_id=sequence, charge_model='amber99', partial_charges=partial_charges)

# if the charges are changed systematically, one could modify grappa.constants.CHARGE_MODELS to include a new charge model, e.g. 'amber99_modified'/'amber99_overcharged'/... and pass this as charge model.

#%%
"""
If we want to train a grappa model, we need to store the dataset as dgl graphs and point the model towards the dataset.
The easiest way to do so is to store the dataset at grappa.utils.dataset_utils.get_data_path()/dgl_datasets/<some_name> and give the model <some_name> as dataset tag. Then, it can be loaded by grappa.data.Dataset.from_tag('<some_name>').
"""

from grappa.data import Dataset
from grappa.utils.dataset_utils import get_data_path

# create dgl graphs from the stored moldata objects:

moldata_paths = list(dspath.glob("moldata_*.npz"))

# loop over the paths, load the moldata objects and convert them to dgl graphs
# also store the mol_id to be able to split the dataset later on
graphs = []
mol_ids = []
for p in tqdm(moldata_paths):
    moldata = MolData.load(p)
    graph = moldata.to_dgl()
    graphs.append(graph)
    mol_ids.append(moldata.mol_id)

#%%
dsname = "example_dataset"

dataset_folder = get_data_path()/'dgl_datasets'/dsname

# create a Grappa Dataset, which stores the graphs, mol_ids and the dataset name
dataset = Dataset(graphs, mol_ids, dsname)

# save the dataset
dataset.save(dataset_folder)

#%%
# now we can load the dataset again from the tag:
dataset = Dataset.from_tag(dsname)
# %%

# we can also train a model on this dataset:
from grappa.training.trainrun import do_trainrun
import yaml

# load the config:
with open(thisdir/"grappa_config.yaml", "r") as f:
    config = yaml.safe_load(f)

config["data_config"]["datasets"] = [dsname] # a 0.8/0.1/0.1 tr/vl/te split is used by default

config["trainer_config"]["max_epochs"] = 50

#%%
do_trainrun(config=config, project="grappa_example_dataset")

"""
In case you want to create a dataset to fine tune grappa for a specific molecule or few molecules and don't want to split the dataset according to the molecule identifier but just according to different conformations, it is best practice to split the dataset yourself and create three datasets for train, val and test set and store them separately with different tags. Then set
config["data_config"]["pure_train_datasets"] = [train_tag]
config["data_config"]["pure_val_datasets"] = [val_tag]
config["data_config"]["pure_test_datasets"] = [test_tag]
"""