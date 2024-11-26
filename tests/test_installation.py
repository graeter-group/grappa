#%%
print("Testing installation...")

from grappa.data import Dataset
from grappa.utils.model_loading_utils import model_from_tag
from grappa.models.energy import Energy
import torch
import copy
#%%
ds = Dataset.from_tag('spice-dipeptide')
ds.create_reference()

model = model_from_tag('latest').eval()
model = torch.nn.Sequential(model, Energy())

#%%


devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']

for device in devices:
    print('----------')
    print(f"Testing on {device}...\n")
    some_dipeptide = copy.deepcopy(ds[42][0])
    some_dipeptide = some_dipeptide.to(device)
    model.to(device)
    some_dipeptide = model(some_dipeptide)

    # assert that the force crmse is below 10 kcal/mol/angstroem
    qm_minus_nonbonded_grad = some_dipeptide.nodes['n1'].data['gradient_ref']
    grappa_grad = some_dipeptide.nodes['n1'].data['gradient']
    crmse = torch.mean((qm_minus_nonbonded_grad - grappa_grad)**2)**0.5
    if crmse > 10:
        raise ValueError(f"Force crmse of a test-molecule is {crmse:3f} kcal/mol/angstroem, which is too high. Something is wrong.")
    
    print(f"Force crmse of a test-molecule is {crmse:3f} kcal/mol/angstroem. Everything seems to be okay.\n\n")
    print('----------')
