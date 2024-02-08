#%%
"""
Evaluate grappas forward pass runtime in cpu mode, dependent on the size of the molecule.
"""
from grappa.data import Dataset
from grappa.utils.loading_utils import model_from_tag
from grappa.data import GraphDataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import copy

device = 'cpu'

device = 'cuda'

#%%
DSNAMES = ['tripeptides_amber99sbildn', 'spice-dipeptide', 'spice-des-monomers', 'rna-diverse']

dataset = sum([Dataset.from_tag(name) for name in DSNAMES], Dataset())

dataset.remove_uncommon_features()

#%%
model = model_from_tag('latest').to(device)


num_atoms = []
runtime = []

#%%
def fill_runtime_data(loader, max_batches=None):
    for i, batch in enumerate(loader):
        if not max_batches is None and i >= max_batches:
            break
        print(f'Batch {i}/{len(loader)}', end='\r')
        g, dsnames = batch

        num_atoms.append(g.num_nodes('n1'))

        with torch.no_grad():
            start = time.time()
            g = g.to(device)
            g_copy = model(g)
            end = time.time()
            runtime.append(end - start)

def plot(num_atoms, runtime, save=False):
    plt.figure()
    plt.xlabel('Number of atoms')
    plt.ylabel('Runtime (s)')
    plt.scatter(num_atoms, runtime, s=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Runtime of grappas forward pass in cpu mode')
    plt.show()
    if save:
        plt.savefig('runtime.png')

#%%
        
if not device == 'cuda':
    loader = GraphDataLoader(copy.deepcopy(dataset), batch_size=2028, shuffle=False, conf_strategy=1)

    fill_runtime_data(loader=loader)
    loader = GraphDataLoader(copy.deepcopy(dataset), batch_size=512, shuffle=False, conf_strategy=1)

    fill_runtime_data(loader=loader)

    loader = GraphDataLoader(copy.deepcopy(dataset), batch_size=256, shuffle=False, conf_strategy=1)

    fill_runtime_data(loader=loader)
    plot(num_atoms, runtime)

loader = GraphDataLoader(copy.deepcopy(dataset), batch_size=128, shuffle=False, conf_strategy=1)

fill_runtime_data(loader=loader)
plot(num_atoms, runtime)

#%%

loader = GraphDataLoader(copy.deepcopy(dataset), batch_size=64, shuffle=False, conf_strategy=1)

fill_runtime_data(loader=loader)
plot(num_atoms, runtime)

loader = GraphDataLoader(copy.deepcopy(dataset), batch_size=32, shuffle=False, conf_strategy=1)

fill_runtime_data(loader=loader)
plot(num_atoms, runtime)

# %%
loader = GraphDataLoader(copy.deepcopy(dataset), batch_size=8, shuffle=False, conf_strategy=1)

fill_runtime_data(loader=loader, max_batches=30)
plot(num_atoms, runtime)

loader = GraphDataLoader(copy.deepcopy(dataset), batch_size=1, shuffle=False, conf_strategy=1)

fill_runtime_data(loader=loader, max_batches=30)
plot(num_atoms, runtime)
# %%

num_atoms_array = np.array(num_atoms)
runtime_array = np.array(runtime)

atom_range = np.logspace(1.2, 6, 500)

import scipy

# fit the data to a + b * n log(n):

def func(x, b, c):
    return b * np.log(x) + c * x * np.log(x)

popt, pcov = scipy.optimize.curve_fit(func, num_atoms_array, runtime_array)

#%%
# popt[0] = 0.02

plt.figure()
plt.xlabel('Number of atoms')
plt.ylabel('Runtime (s)')
plt.scatter(num_atoms, runtime, s=10)
plt.xscale('log')
plt.yscale('log')
plt.plot(atom_range, func(atom_range, *popt), color='red', linestyle='--', label=rf'fit: t = {popt[0]:.2f} log(n) + {popt[1]:.1e} n log(n)')
plt.legend()

if 'cuda' in device:
    plt.title('Runtime of Grappas Forward Pass (GPU)')
    plt.savefig('runtime_gpu.png', dpi=500)
else:
    plt.title('Runtime of Grappas Forward Pass (CPU)')
    plt.savefig('runtime.png', dpi=500)
plt.show()
# %%
func(3e5, *popt)
# %%
