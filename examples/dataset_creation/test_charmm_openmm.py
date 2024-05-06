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


from openmm.app import PDBFile, ForceField, Modeller

pdb = PDBFile("tripeptide_example_data/pdb_0.pdb")
topology = pdb.getTopology()
# model the pdb:
modeller = Modeller(topology, pdb.getPositions())
# load the forcefield:
forcefield = ForceField('amber99sbildn.xml')
# add the forcefield to the modeller:
modeller.addForceField(forcefield)

# create the system:
system = forcefield.createSystem(modeller.topology)

# %%
