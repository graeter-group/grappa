#%%
# STANDARD OPENMM WORKFLOW
######################
from pathlib import Path
from openmm.app import ForceField, Topology, PDBFile
from openmm.app import Modeller
from openmm import unit
from grappa import OpenmmGrappa
from grappa.utils import get_repo_dir

thisdir = Path(__file__).parent

# pdbpath = get_repo_dir()/'examples/dataset_creation/tripeptide_example_data/pdb_0.pdb'
pdbpath = get_repo_dir()/'examples/usage/T4.pdb'
pdbpath = str(pdbpath)

pdbfile = PDBFile(pdbpath)
topology = pdbfile.topology # load your system as openmm.Topology

tip_3p_path = get_repo_dir()/'src/grappa/utils/classical_forcefields/tip3p_standard.xml'
tip_3p_path = str(tip_3p_path)

# classical_ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
classical_ff = ForceField('/hits/fast/mbm/hartmaec/workdir/FF99SBILDNPX_OpenMM/grappa_1-3-amber99_ff99SB.xml', tip_3p_path)
# solvate:

modeller = Modeller(topology, pdbfile.positions)
modeller.deleteWater()
# modeller.addHydrogens(classical_ff)
# modeller.addSolvent(classical_ff, model='tip3p', padding=1.0*unit.nanometers)

topology = modeller.getTopology()
positions = modeller.getPositions()

system_grappa_tab = classical_ff.createSystem(topology)
##########################

#%%
import copy
# load the pretrained ML model from a tag. Currently, possible tags are grappa-1.3' and 'latest'
grappa_ff = OpenmmGrappa.from_tag('grappa-1.3')

# grappa will not change the solvant parameters and the nonbonded parameters, e.g. the partial charges, Lennard-Jones parameters and combination rules
system_grappa = grappa_ff.parametrize_system(copy.deepcopy(system_grappa_tab), topology, plot_dir=thisdir)

system_amber = ForceField('amber99sbildn.xml', 'tip3p.xml').createSystem(topology)

# %%

# SMALL VALIDATION

# now we can use this system downstream. To validate that grappa predicts gradients that are somewhat comparable to those of the classical protein force field, we can plot the gradient components of the grappa system and the original system:

from grappa.utils.openmm_utils import get_energies
import numpy as np
from grappa.constants import get_grappa_units_in_openmm
from grappa.utils.plotting import scatter_plot

DISTANCE_UNIT = get_grappa_units_in_openmm()['LENGTH']
positions = np.array([positions.value_in_unit(DISTANCE_UNIT)])

# # get energies and gradients of the original system:
orig_energy, original_gradients = get_energies(system_grappa_tab, positions)

grappa_energy, grappa_gradients = get_energies(system_grappa, positions)

# %%

from matplotlib import pyplot as plt

fig, ax = plt.subplots()
ax = scatter_plot(ax=ax, x=original_gradients.flatten(), y=grappa_gradients.flatten(), cluster=True, show_rmsd=True, logscale=True)
ax.set_xlabel('Grapa-tab')
ax.set_ylabel('Grappa')
ax.set_title('Gradients [kcal/mol/A]')

fig.savefig(str(thisdir/'grappa_vs_classical_gradients_T4.png'))
print('Saved fig to grappa_vs_classical_gradients.png')
fig.show()
#%%


from grappa.data import Parameters, Molecule

mol = Molecule.from_openmm_system(system_grappa_tab, topology)
params_amber = Parameters.from_openmm_system(system_amber, mol)

params_grappa_tab = Parameters.from_openmm_system(system_grappa_tab, mol)
# %%
params_grappa_tab.compare_with(params_amber, filename=thisdir/'parameter_comparison_tab_amber.png', xlabel='Grappa', ylabel='Amber')
# %%
