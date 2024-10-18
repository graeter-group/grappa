#%%
# STANDARD OPENMM WORKFLOW
######################
from pathlib import Path
from openmm.app import ForceField, Topology, PDBFile
from openmm.app import Modeller
from openmm import unit
from grappa import OpenmmGrappa

thisdir = Path(__file__).parent
pdbfile = PDBFile(str(thisdir/'T4.pdb'))
topology = pdbfile.topology # load your system as openmm.Topology

classical_ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
# solvate:

modeller = Modeller(topology, pdbfile.positions)
modeller.deleteWater()
modeller.addHydrogens(classical_ff)
modeller.addSolvent(classical_ff, model='tip3p', padding=1.0*unit.nanometers)

topology = modeller.getTopology()
positions = modeller.getPositions()

system = classical_ff.createSystem(topology)
##########################

#%%
# load the pretrained ML model from a tag. Currently, possible tags are grappa-1.3' and 'latest'
grappa_ff = OpenmmGrappa.from_tag('latest')

# grappa will not change the solvant parameters and the nonbonded parameters, e.g. the partial charges, Lennard-Jones parameters and combination rules
system = grappa_ff.parametrize_system(system, topology, plot_dir=thisdir)

# %%

# SMALL VALIDATION

# now we can use this system downstream. To validate that grappa predicts gradients that are somewhat comparable to those of the classical protein force field, we can plot the gradient components of the grappa system and the original system:

orig_system = classical_ff.createSystem(topology)
from grappa.utils.openmm_utils import get_energies
import numpy as np
from grappa.constants import get_grappa_units_in_openmm
from grappa.utils.plotting import scatter_plot

DISTANCE_UNIT = get_grappa_units_in_openmm()['LENGTH']
positions = np.array([positions.value_in_unit(DISTANCE_UNIT)])

# # get energies and gradients of the original system:
orig_energy, original_gradients = get_energies(orig_system, positions)

grappa_energy, grappa_gradients = get_energies(system, positions)

# %%

from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax = scatter_plot(ax=ax, x=original_gradients.flatten(), y=grappa_gradients.flatten(), cluster=True, show_rmsd=True, logscale=True)
ax.set_xlabel('FF99SBILDN')
ax.set_ylabel('Grappa')
ax.set_title('Gradients [kcal/mol/A]')

fig.savefig(str(thisdir/'grappa_vs_classical_gradients_T4.png'))
print('Saved fig to grappa_vs_classical_gradients.png')
fig.show()
# %%
