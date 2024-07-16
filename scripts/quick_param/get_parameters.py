#%%
# STANDARD OPENMM WORKFLOW
######################
from pathlib import Path
from openmm.app import ForceField, Topology, PDBFile
from openmm.app import Modeller
from openmm import unit
from grappa import Grappa
from grappa.utils import get_repo_dir
import copy
from grappa.data import Molecule, Parameters
from grappa.data.parameters import compare_parameters, scatter_plot
from grappa.utils.plotting import calculate_density_scatter

thisdir = Path(__file__).parent

# pdbpath = get_repo_dir()/'examples/dataset_creation/tripeptide_example_data/pdb_0.pdb'
pdbpath = get_repo_dir()/'examples/usage/T4.pdb'
pdbpath = str(pdbpath)

pdbfile = PDBFile(pdbpath)
topology = pdbfile.topology # load your system as openmm.Topology

#%%

modeller = Modeller(topology, pdbfile.positions)
modeller.deleteWater()

topology = modeller.getTopology()

system = ForceField("amber99sbildn.xml").createSystem(topology)
##########################
# %%

grappa_ff = Grappa.from_tag('grappa-1.3')

molecule = Molecule.from_openmm_system(openmm_system=system, openmm_topology=topology)

reference_parameters = copy.deepcopy(Parameters.from_openmm_system(openmm_system=system, mol=molecule, allow_skip_improper=True))

#%%
# predict parameters
parameters = grappa_ff.predict(molecule)
# %%



data = compare_parameters(reference_parameters, parameters, xlabel='ff99SB-ILDN', ylabel='Grappa-1.3', s=50, get_values=True)
# %%

# Units dictionary
UNITS = {
    'bond_eq': r'$\AA$',
    'bond_k': r'kcal/mol/$\AA^2$',
    'angle_eq': 'deg',
    'angle_k': r'kcal/mol/deg$^2$',
    'proper_ks': r'kcal/mol',
    'improper_ks': r'kcal/mol',
}

n_periodicity = 3

TITLES = [
    f'Bond eq. [{UNITS["bond_eq"]}]',
    f'Bond k [{UNITS["bond_k"]}]',
    f'Angle eq. [{UNITS["angle_eq"]}]',
    f'Angle k [{UNITS["angle_k"]}]',
    f'Torsion k (n=1) [{UNITS["proper_ks"]}]',
    f'Torsion k (n=2-{n_periodicity}) [{UNITS["proper_ks"]}]'
]

import numpy as np
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

ff99_data = data['x']
grappa_data = data['y']

#%%

font="Arial"
fontsize=16
titlesize=fontsize+3

import matplotlib.pyplot as plt

plt.rc('font', family=font)
plt.rc('font', size=fontsize)
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
plt.rc('axes', titlesize=titlesize)
plt.rc('axes', labelsize=fontsize+1)
plt.rc('legend', fontsize=fontsize)

kwargs = {
    's': 50,
}

keys = [
    f'Bond eq.',
    f'Bond k',
    f'Angle eq.',
    f'Angle k',
    f'Torsion k (n=1)',
    f'Torsion k (n=2-{n_periodicity})'
]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

idxs = [0,1,2,3]


axs = axs.flatten()
max_freq = 0
for idx, ax in zip(idxs, axs):
    grappa_vals = grappa_data[keys[idx]]
    ff99_vals = ff99_data[keys[idx]]
    points, frequencies = calculate_density_scatter(ff99_vals, grappa_vals, delta_factor=60)
    max_freq = max(max_freq, max(frequencies))


for idx, ax in zip(idxs, axs):

    title = TITLES[idx]
    grappa_vals = grappa_data[keys[idx]]
    ff99_vals = ff99_data[keys[idx]]
    points, frequencies = calculate_density_scatter(ff99_vals, grappa_vals, delta_factor=60)

    scatter = ax.scatter(points[:,0], points[:,1], norm=colors.LogNorm(vmin=1, vmax=max_freq), c=frequencies, cmap='viridis', edgecolor='k', linewidths=0.8, **kwargs)

    ax.set_title(title)

    if idx > 1:
        ax.set_xlabel('ff99SB-ILDN')
    if idx % 2 == 0:
        ax.set_ylabel('Grappa-1.3')
    ax.set_aspect('equal')
    
    ax.tick_params(axis='both', which='major', direction='inout', length=10, width=1)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(-3,4))


    x_min, y_min = ax.get_xlim()[0], ax.get_ylim()[0]
    x_max, y_max = ax.get_xlim()[1], ax.get_ylim()[1]

    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)

    # reference line
    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='-', linewidth=0.5)

    ax.set_ylim(min_val, max_val)
    ax.set_xlim(min_val, max_val)



# Create an axis for the colorbar
cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Adjust these values to position your colorbar
cbar = fig.colorbar(scatter, cax=cbar_ax)
# cbar.set_label('Frequency')

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust subplot params to fit the colorbar
plt.show()
plt.savefig('param_compare.png', dpi=300)
# %%
