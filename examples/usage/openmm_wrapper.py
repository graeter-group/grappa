#%%
# USAGE

# Download a model if not present already:
import torch
from grappa.utils.loading_utils import model_from_tag

model = model_from_tag('latest')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

#%%

from grappa.wrappers.openmm_wrapper import openmm_Grappa
from openmm.app import PDBFile, ForceField
from copy import deepcopy
from pathlib import Path
import time
#%%

pdb = PDBFile(str(Path(__file__).parent/'T4.pdb'))

print(f'Loaded a protein with {pdb.topology.getNumAtoms()} atoms.')

ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
start = time.time()
system = ff.createSystem(pdb.topology)
print(f'Created a system in {(time.time()-start)*1e3:.2f} milliseconds')

orig_system = deepcopy(system) # create a deep copy of the system to compare it to the grappa-parametrized one later

# %%

# build a grappa model that handles the ML pipeline
grappa = openmm_Grappa(model, device=device)

# write grappa parameters to the system:
start = time.time()
system = grappa.parametrize_system(system, pdb.topology)
print(f'Parametrized the system in {(time.time()-start)*1e3:.2f} milliseconds')

# %%

# SMALL VALIDATION

# now we can use this system downstream. To validate that grappa predicts gradients that are somewhat comparable to those of the classical protein force field, we can plot the gradient components of the grappa system and the original system:
from grappa.utils.openmm_utils import get_energies
import numpy as np
from grappa.units import DISTANCE_UNIT

positions = np.array([pdb.positions.value_in_unit(DISTANCE_UNIT)])

# get energies and gradients of the original system:
orig_energy, original_gradients = get_energies(orig_system, positions)

grappa_energy, grappa_gradients = get_energies(system, positions)
# %%

from matplotlib import pyplot as plt

plt.scatter(original_gradients.flatten(), grappa_gradients.flatten())
plt.xlabel('original gradients')
plt.ylabel('grappa gradients')
plt.title('Gradients [kcal/mol/A]')

crmse = np.sqrt(np.mean((original_gradients.flatten() - grappa_gradients.flatten())**2))
plt.text(0.1, 0.9, f'Component RMSE: {crmse:.2f} kcal/mol/A', transform=plt.gca().transAxes)

plt.plot(original_gradients.flatten(), original_gradients.flatten(), color='black', linestyle='--')

plt.show()
# %%
