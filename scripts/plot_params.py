#%%
from grappa.data import MolData
from grappa.data.Parameters import plot_parameters
from pathlib import Path

dspath = Path(__file__).parent.parent / 'data' / 'grappa_datasets' / 'spice-dipeptide_amber99sbildn'

params = []
qm_forces = []
amber_forces = []

for i in range(10):
    try:
        data = MolData.load(f'{dspath}/{i}.npz')

        params.append(data.classical_parameters)
    except:
        pass

fig,axes = plot_parameters(params)

fig.savefig('spice-dipeptide_amber99sbildn.png')