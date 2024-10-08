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

# grappa_tab_ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
grappa_tab_ff = ForceField('/hits/fast/mbm/hartmaec/workdir/FF99SBILDNPX_OpenMM/grappa_1-3-amber99_ff99SB.xml', tip_3p_path)
# solvate:

modeller = Modeller(topology, pdbfile.positions)
modeller.deleteWater()
modeller.addHydrogens(grappa_tab_ff)
modeller.addSolvent(grappa_tab_ff, model='tip3p', padding=1.0*unit.nanometers)

topology = modeller.getTopology()
positions = modeller.getPositions()

system_grappa_tab = grappa_tab_ff.createSystem(topology)
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

params_grappa_tab = Parameters.from_openmm_system(system_grappa_tab, mol)
#%%
def k_hist(params, name=''):
    proper_ks = params.proper_ks

    # print(proper_ks.min(axis=0))
    # print(proper_ks.max(axis=0))
    # print(proper_ks.mean(axis=0))
    # print(proper_ks.std(axis=0))

    h = plt.hist(proper_ks[:,0], bins=100)
    plt.title(f'{name} Proper k n=1')
    plt.yscale('log')
    plt.savefig(str(thisdir/f'{name}_proper_k_1.png'))
    plt.show()

    h = plt.hist(proper_ks[:,1], bins=100)
    plt.title(f'{name} Proper k n=2')
    plt.yscale('log')
    plt.savefig(str(thisdir/f'{name}_proper_k_2.png'))
    plt.show()

    h = plt.hist(proper_ks[:,2], bins=100)
    plt.title(f'{name} Proper k n=3')
    plt.yscale('log')	
    plt.savefig(str(thisdir/f'{name}_proper_k_3.png'))
    plt.show()

k_hist(params_grappa_tab, name='grappa_tab')

# %%
params_grappa = Parameters.from_openmm_system(system_grappa, mol)
k_hist(params_grappa, name='grappa')
params_grappa_tab.compare_with(params_grappa, filename=thisdir/'parameter_comparison_tab_grappa.png', xlabel='Grappa tab', ylabel='grappa')
# %%
