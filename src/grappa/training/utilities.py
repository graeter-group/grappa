import os
import sys

from ..models.energy import WriteEnergy

import torch
import numpy as np


def get_bonded_parameter_types():
    return [("n2", "k"), ("n2", "eq"), ("n3", "k"), ("n3", "eq"), ("n4", "k"), ("n4_improper", "k")]


def get_param_statistics(loader, bonded_parameter_types=get_bonded_parameter_types()):
    class_ff = "ref"
    with torch.no_grad():
        params = bonded_parameter_types
        d = {"mean":{}, "std":{}}
        for level, name in params:
            all_params = torch.cat([g.nodes[level].data[f"{name}_{class_ff}"]
                                    if level in g.ntypes else torch.tensor([])
                                    for g in loader], dim=0)
            if len(all_params) == 0:
                # do not write the statistics for this parameter type, we cannot predict it anyways if it is not in the train set.
                continue
            d["mean"][f"{level}_{name}"] = all_params.mean(dim=0)
            d["std"][f"{level}_{name}"] = min(all_params.std(dim=0), 1e-6) # minimum std deviation: 1e-6
    return d

def get_default_statistics():
    return get_default_statistics.d

get_default_statistics.d = {
'mean':
    {'n2_k': torch.Tensor([763.2819]), 'n2_eq': torch.Tensor([1.2353]), 'n3_k': torch.Tensor([105.6576]), 'n3_eq': torch.Tensor([1.9750]), 'n4_k': torch.Tensor([ 1.5617e-01, -5.8312e-01,  7.0820e-02, -6.3840e-04,  4.7139e-04, -4.1655e-04]), 'n4_improper_k': torch.Tensor([ 0.0000, -2.3933,  0.0000,  0.0000,  0.0000,  0.0000])},
'std':
    {'n2_k': torch.Tensor([161.2278]), 'n2_eq': torch.Tensor([0.1953]), 'n3_k': torch.Tensor([26.5965]), 'n3_eq': torch.Tensor([0.0917]), 'n4_k': torch.Tensor([0.4977, 1.2465, 0.1466, 0.0192, 0.0075, 0.0066]), 'n4_improper_k': torch.Tensor([0.0000, 4.0571, 0.0000, 0.0000, 0.0000, 0.0000])}
}



def shape_test(ds, n_check=1, force_factor=0):

    assert len(ds) > 0

    for i in np.random.permutation(len(ds))[:n_check]:
        assert ds[i].nodes["g"].data["u_qm"].shape[1] == ds[i].nodes["n1"].data["xyz"].shape[1], f"u_qm: {ds[i].nodes['g'].data['u_qm'].shape}, u_ref: {ds[i].nodes['g'].data['u_ref'].shape}"

        assert ds[i].nodes["g"].data["u_qm"].shape == ds[i].nodes["g"].data["u_ref"].shape, f"u_qm: {ds[i].nodes['g'].data['u_qm'].shape}, u_ref: {ds[i].nodes['g'].data['u_ref'].shape}"

        if force_factor > 0:
            assert ds[i].nodes["n1"].data["xyz"].shape == ds[i].nodes["n1"].data["grad_ref"].shape, f"xyz: {ds[i].nodes['n1'].data['xyz'].shape}, grad_ref: {ds[i].nodes['n1'].data['grad_ref'].shape}"

        assert ds[i].nodes["n2"].data["idxs"].dtype in [torch.long, torch.int32], f"idxs dtype: {ds[i].nodes['n2'].data['idxs'].dtype}"


def get_grad(model, batch, device="cpu", energy_writer=WriteEnergy(), retain_graph=False):

    # recognize a possible context in which the function is called
    def grad_active():
        x = torch.tensor([1.], requires_grad=True)
        y = x * 2
        return y.requires_grad # is false if context is torch.no_grad()
    
    if not grad_active():
        raise RuntimeError("In get_grad: context does not allow the calculation of gradients")


    batch.nodes["n1"].data["xyz"].requires_grad = True
    batch = batch.to(device)
    model = model.to(device)
    batch = model(batch)
    batch = energy_writer(batch)
    energies = batch.nodes["g"].data["u"]

    # omit the minus sign
    grad = torch.autograd.grad(energies.sum(), batch.nodes["n1"].data["xyz"], retain_graph=True, create_graph=retain_graph)[0]

    return grad, model, batch



