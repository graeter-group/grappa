#%%
from grappa.data import MolData
from grappa.wrappers import openmm_wrapper
from grappa.utils import model_loading_utils

#%%

molpath1 = '/hits/fast/mbm/seutelf/grappa/data/grappa_datasets/spice-dipeptide_amber99sbildn/3.npz'

molpath2 = '/hits/fast/mbm/seutelf/grappa/data/grappa_datasets/spice-dipeptide/3.npz'

molpath3 = '/hits/fast/mbm/seutelf/grappa/data/grappa_datasets/tripeptides_amber99sbildn/3.npz'

mol1 = MolData.load(molpath1)
mol2 = MolData.load(molpath2)

#%%

print('In Grappa ds:')
print(mol1.molecule.additional_features['charge_model'].mean(0))
print(mol2.molecule.additional_features['charge_model'].mean(0))
# %%
# load the dgl datasets:
from grappa.data import Dataset

ds1 = Dataset.from_tag('spice-dipeptide_amber99sbildn')
ds2 = Dataset.from_tag('spice-dipeptide')
ds3 = Dataset.from_tag('tripeptides_amber99sbildn')

g1 = ds1[0][0]
g2 = ds2[0][0]
g3 = ds3[0][0]

print('In DGL ds:')
print(g1.nodes['n1'].data['charge_model'].mean(0))
print(g2.nodes['n1'].data['charge_model'].mean(0))
print(g3.nodes['n1'].data['charge_model'].mean(0))

#%%

from grappa.utils.openmm_utils import topology_from_pdb
from grappa.data import Molecule
from openmm.app import ForceField

top = topology_from_pdb(MolData.load(molpath3).pdb)

ff = ForceField('amber99sbildn.xml')

system = ff.createSystem(top)

mol1 = Molecule.from_openmm_system(openmm_system=system, openmm_topology=top, charge_model='classical')

print(mol1.additional_features['charge_model'].mean(0))

mol2 = Molecule.from_openmm_system(openmm_system=system, openmm_topology=top, charge_model='am1BCC')

print(mol2.additional_features['charge_model'].mean(0))

# %%
from grappa.utils.model_loading_utils import model_from_tag
from grappa.models import Energy
import torch

model = model_from_tag('grappa-1.0')

model = torch.nn.Sequential(model, Energy())
# %%
g = ds1[0][0]

with torch.no_grad():
    g = model(g)

grads = g.nodes['n1'].data['gradient']
ref_grads = g.nodes['n1'].data['gradient_ref']

print(torch.sqrt(((grads - ref_grads)**2).mean()))
# %%
ds = Dataset.from_tag('tripeptides_amber99sbildn')

len(ds)
#%%
g = ds[0][0]

grad_classical = g.nodes['n1'].data['gradient_reference_ff']
grad_qm = g.nodes['n1'].data['gradient_qm']

print('on tripeptides dataset:')
print(g.nodes['n1'].data['charge_model'].mean(0))
print(torch.sqrt(((grad_classical - grad_qm)**2).mean()))
# %%
# %%
g.nodes['n1'].data['partial_charge']
# %%
