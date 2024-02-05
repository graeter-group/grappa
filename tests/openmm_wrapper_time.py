#%%
# USAGE

# Download a model if not present already:
import torch
from grappa.utils.loading_utils import model_from_tag

model = model_from_tag('latest')

device = 'cpu'

model = model.to(device)

#%%

from grappa.wrappers.openmm_wrapper import openmm_Grappa
from openmm.app import PDBFile, ForceField
from copy import deepcopy
from pathlib import Path
import time
#%%

pdb = PDBFile(str(Path(__file__).parent.parent/'examples/usage/T4.pdb'))

print(f'Loaded a protein with {pdb.topology.getNumAtoms()} atoms.')

ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
start = time.time()
system = ff.createSystem(pdb.topology)
print(f'Created a system in {(time.time()-start):.3f} seconds')

# %%

# build a grappa model that handles the ML pipeline
grappa = openmm_Grappa(model, device=device)

# write grappa parameters to the system:
start = time.time()
system = grappa.parametrize_system(system, pdb.topology)
print(f'Parametrized the system in {(time.time()-start):.3f} seconds')

# %%

import cProfile

def profile_function():
    pdb = PDBFile(str(Path(__file__).parent.parent/'examples/usage/T4.pdb'))

    print(f'Loaded a protein with {pdb.topology.getNumAtoms()} atoms.')

    ff = ForceField('amber99sbildn.xml', 'tip3p.xml')
    start = time.time()
    system = ff.createSystem(pdb.topology)
    print(f'Created a system in {(time.time()-start):.3f} seconds')

    system = grappa.parametrize_system(system, pdb.topology)
    

# Create a profiler object
profiler = cProfile.Profile()

# Run the function within the context of the profiler
profiler.runcall(profile_function)

# Print the statistics from the profiler
profiler.print_stats()
# %%
