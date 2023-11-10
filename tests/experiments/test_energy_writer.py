#%%
import dgl
import torch
from grappa.models import get_models
from pathlib import Path
from grappa.utils.dgl_utils import batch, unbatch

#%%
class ParamFixer(torch.nn.Module):
    def forward(self, g):
        g.nodes['n2'].data['k'] = g.nodes['n2'].data['k'][:,0]
        g.nodes['n2'].data['eq'] = g.nodes['n2'].data['eq'][:,0]
        g.nodes['n3'].data['k'] = g.nodes['n3'].data['k'][:,0]
        g.nodes['n3'].data['eq'] = g.nodes['n3'].data['eq'][:,0]
        return g

model = torch.nn.Sequential(
    get_models.get_full_model(in_feat_name=["atomic_number"]),
    ParamFixer()
)
dglpath = Path(__file__).parents[2]/'data'/'dgl_datasets'
# %%
dsname = 'spice-dipeptide'
ds, _ = dgl.load_graphs(str(dglpath/dsname)+".bin")

num_confs = []
for g in ds:
    n = len(g.nodes['g'].data['energy_ref'][0])
    num_confs.append(n)
min(num_confs)

# make all graphs have the same number of conformations and batch them together
for g in ds:
    g.nodes['n1'].data['xyz'] = g.nodes['n1'].data['xyz'][:,:min(num_confs)]
    g.nodes['g'].data['energy_ref'] = g.nodes['g'].data['energy_ref'][:,:min(num_confs)]
    for feat in g.nodes['g'].data.keys():
        if 'energy' in feat:
            g.nodes['g'].data[feat] = g.nodes['g'].data[feat][:,:min(num_confs)]
    for feat in g.nodes['n1'].data.keys():
        if 'gradient' in feat:
            g.nodes['n1'].data[feat] = g.nodes['n1'].data[feat][:,:min(num_confs)]


batchsize = 3
batches = []
for i in range(0, len(ds), batchsize):
    batches.append(batch(ds[i:min(i+batchsize, len(ds))]))


tr, vl, te = dgl.data.utils.split_dataset(batches, [0.8, 0.1, 0.1])

#%%
mse = torch.nn.MSELoss()
def parameter_loss(g):
    loss = 0
    loss += mse(g.nodes['n2'].data['k_ref'], g.nodes['n2'].data['k'])/1000**2
    loss += mse(g.nodes['n2'].data['eq_ref'], g.nodes['n2'].data['eq'])
    loss += mse(g.nodes['n3'].data['k_ref'], g.nodes['n3'].data['k'])/100**2
    loss += mse(g.nodes['n3'].data['eq_ref'], g.nodes['n3'].data['eq'])
    loss += mse(g.nodes['n4'].data['k_ref'], g.nodes['n4'].data['k'])
    return loss

# redefine parameter loss to act on tuple energy contributions
from grappa.models.Energy import Energy
ref_writer = Energy(terms=['n2', 'n3', 'n4','n4_improper'], write_suffix="_classical", gradients=False, offset_torsion=False, suffix="_ref")

pred_writer = Energy(terms=['n2', 'n3', 'n4','n4_improper'], write_suffix="", gradients=False, offset_torsion=True, suffix="")

def tuplewise_loss(g):
    se = 0
    n_confs = g.nodes['g'].data['energy_ref'].shape[1]
    n_batches = g.nodes['g'].data['energy_ref'].shape[0]
    for term in ['n2', 'n3', 'n4','n4_improper']:
        if term in g.ntypes:
            se += torch.square(g.nodes[term].data['energy_classical'] - g.nodes[term].data['energy']).sum()
    se_per_conf = se/(n_confs*n_batches)
    return se_per_conf

#%%
device='cuda'
def do_eval(vl:list, model, epoch):
    model.eval()
    model = model.to(device)
    vl_loss = 0
    vl_confs = 0
    vl_en_rmse = 0
    vl_grad_crmse = 0
    for i, g in enumerate(vl):
        g = g.to(device)
        g = model(g)
        g = pred_writer(g)
        g = ref_writer(g)
        loss = parameter_loss(g)
        num_confs = len(g.nodes['g'].data['energy_ref'][0])
        vl_confs += num_confs
        vl_loss += loss.detach() # calculate the total se instead of the mean
        vl_en_rmse += torch.sqrt(mse(g.nodes['g'].data['energy_classical'], g.nodes['g'].data['energy'])).detach()
        # vl_grad_crmse += torch.sqrt(mse(g.nodes['n1'].data['gradient_classical'], g.nodes['n1'].data['gradient'])).detach()

    print(f"Epoch {epoch}, mean vl parameter loss {vl_loss.item()/len(vl)}, mean vl en_classical rmse {vl_en_rmse.item()/len(vl)}\n")
#%%
device='cuda'
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# gradient clipping:
torch.nn.utils.clip_grad_norm_(model.parameters(), 1e2)
do_eval(vl, model, -1)
for epoch in range(5):
    total_loss = 0
    total_confs = 0
    total_atoms = 0
    for i, g in enumerate(tr):
        optimizer.zero_grad()
        g = g.to(device)
        g = model(g)
        g = ref_writer(g)
        g = pred_writer(g)
        loss = parameter_loss(g)*1e2
        loss += tuplewise_loss(g)
        loss.backward()
        optimizer.step()

        num_confs = len(g.nodes['g'].data['energy_ref'][0])
        num_atoms = len(g.nodes['n1'].data['xyz'])
        total_confs += num_confs
        total_loss += loss.detach() # calculate the total se instead of the mean
        if i % 100 == 0:
            print(f"Epoch {epoch}, instance {i}, mean tr loss {total_loss.item()/(i+1)}")
    do_eval(vl, model, 0)


#%%

g = ds[0]
g.nodes['n2'].data['energies'] = torch.randn((g.num_nodes("n2"), g.nodes['g'].data['energy_ref'].shape[1]))
print(g.nodes['n2'].data['energies'].shape)
print(g.nodes['g'].data['energy_ref'].shape)
print(dgl.readout_nodes(g, op='sum', ntype='n2', feat='energies').shape)

print()

g = tr[0]
g.nodes['n2'].data['energies'] = torch.randn((g.num_nodes("n2"), g.nodes['g'].data['energy_ref'].shape[1]))
print(g.nodes['n2'].data['energies'].shape)
print(g.nodes['g'].data['energy_ref'].shape)
print(dgl.readout_nodes(g, op='sum', ntype='n2', feat='energies').shape)
#%%


from grappa.models.Energy import Energy

######################

improper_energy_writer = Energy(terms=['n4_improper'], write_suffix="_improper", gradients=True)

def improper_energy_mse(g):
    loss = 0
    improper_contrib_pred = g.nodes['g'].data['energy_improper']
    improper_contrib_ref = g.nodes['g'].data['improper_energy_ref']

    improper_contrib_pred = improper_contrib_pred - improper_contrib_pred.mean(dim=1, keepdim=True)
    improper_contrib_ref -= improper_contrib_ref.mean(dim=1, keepdim=True)

    loss += mse(improper_contrib_pred, improper_contrib_ref)
    return loss

def improper_grad_mse(g):
    loss = 0
    improper_contrib_pred = g.nodes['n1'].data['gradient_improper']
    improper_contrib_ref = g.nodes['n1'].data['improper_gradient_ref']

    loss += mse(improper_contrib_pred, improper_contrib_ref)
    return loss

def energy_mse(g):
    loss = 0
    energy_pred = g.nodes['g'].data['energy']
    energy_ref = g.nodes['g'].data['energy_ref']

    energy_pred = energy_pred - energy_pred.mean(dim=1, keepdim=True)
    energy_ref -= energy_ref.mean(dim=1, keepdim=True)

    loss += mse(energy_pred, energy_ref)
    return loss

def gradient_mse(g):
    loss = 0
    grad_pred = g.nodes['n1'].data['gradient']
    grad_ref = g.nodes['n1'].data['gradient_ref']

    loss += mse(grad_pred, grad_ref)
    return loss


#%%
g = tr[10]
g = g.to(device)
g = model(g)

energy_writer = Energy(terms=['n2', 'n3', 'n4', "n4_improper"], write_suffix="", gradients=True, suffix="_ref", offset_torsion=False)

g = energy_writer(g)
g = ref_writer(g)
#%%
# import matplotlib.pyplot as plt
# plt.scatter(g.nodes['n1'].data['gradient'].flatten().cpu().numpy(), g.nodes['n1'].data['gradient_classical'].flatten().cpu().numpy())
# plt.show()

energies = g.nodes['g'].data['energy']# - g.nodes['g'].data['energy'].mean(dim=1, keepdim=True)
energies_ref = g.nodes['g'].data['energy_classical']# - g.nodes['g'].data['energy_classical'].mean(dim=1, keepdim=True)

plt.scatter(energies.flatten().cpu().detach().numpy(), energies_ref.flatten().cpu().detach().numpy())

#%%
energies.shape
print(g.nodes['g'].data['energy_n3'].std(dim=1))
g.nodes['n2'].data['x'].mean(dim=-1)
#%%

# %%
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for epoch in range(2):
    vl_loss = 0
    vl_confs = 0
    vl_en_mse = 0
    vl_grad_mse = 0
    for i, g in enumerate(vl):
        g = g.to(device)
        g = model(g)
        g = improper_energy_writer(g)
        en = improper_energy_mse(g)
        grad = improper_grad_mse(g)
        vl_en_mse += en.detach()
        vl_grad_mse += grad.detach()
    print(f"Epoch {epoch}, mean vl en mse {vl_en_mse.item()/len(vl)}, mean vl grad mse {vl_grad_mse.item()/len(vl)}\n")


    total_loss = 0
    total_confs = 0
    total_atoms = 0
    en_mse = 0
    grad_mse = 0
    for i, g in enumerate(tr):
        optimizer.zero_grad()
        g = g.to(device)
        g = model(g)
        g = improper_energy_writer(g)
        en = improper_energy_mse(g)
        grad = improper_grad_mse(g)
        loss = parameter_loss(g)*1e-2 + en + grad
        loss.backward()
        optimizer.step()

        num_confs = len(g.nodes['g'].data['energy_ref'][0])
        num_atoms = len(g.nodes['n1'].data['xyz'])
        total_confs += num_confs
        total_loss += loss.detach()
        en_mse += en.detach()
        grad_mse += grad.detach()
        if i % 100 == 0:
            print(f"Epoch {epoch}, instance {i}, mean en mse {en_mse.item()/(i+1)}, mean grad mse {grad_mse.item()/(i+1)}")


        

#%%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

energy_writer = Energy(terms=['n2', 'n3', 'n4', "n4_improper"], write_suffix="", gradients=True)

do_eval(vl, model, -1)
for epoch in range(10):

    total_loss = 0
    total_confs = 0
    total_atoms = 0
    en_mse = 0
    grad_mse = 0
    for i, g in enumerate(tr):
        optimizer.zero_grad()
        g = g.to(device)
        g = model(g)
        g = energy_writer(g)
        en = energy_mse(g)
        grad = gradient_mse(g)
        loss = parameter_loss(g)*1e-2 + en + grad
        loss.backward()
        optimizer.step()

        num_confs = len(g.nodes['g'].data['energy_ref'][0])
        num_atoms = len(g.nodes['n1'].data['xyz'])
        total_confs += num_confs
        total_loss += loss.detach()
        en_mse += en.detach()
        grad_mse += grad.detach()
        if i % 100 == 0:
            print(f"Epoch {epoch}, instance {i}, mean en mse {en_mse.item()/(i+1)}, mean grad mse {grad_mse.item()/(i+1)}")

    do_eval(vl, model, epoch)

        
#%%
g = ds[0]

g = g.to(device)
g = model(g)
g = energy_writer(g)

# %%
from matplotlib import pyplot as plt
plt.scatter(g.nodes['n1'].data['gradient'].flatten().cpu().numpy(), g.nodes['n1'].data['gradient_ref'].flatten().cpu().numpy())
plt.show()
plt.scatter(g.nodes['g'].data['energy'].flatten().cpu().detach().numpy(), g.nodes['g'].data['energy_ref'].flatten().cpu().detach().numpy())
plt.show()
# %%
