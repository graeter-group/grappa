"""
A collection of loss functions for training grappa models.
"""

import dgl
import torch
from typing import Dict, List
from grappa.utils import graph_utils


class MolwiseLoss(torch.nn.Module):
    """
    Unbatch the graph and calculate the mse averaged over spatial dimension, conformation and atom. Then average these over the batch. Note: The averages do not commute because the number of atoms in the graph may differ. This way we assign the same weight to molecules with different amounts of atoms.
    If skip_params_if_not_present is set to True, the parameters are only included in the loss if they are present in the graph. If a parameter is present but not a number, the loss contribution from these parameters is zero. Thus, if you have a dataset where some molecules have parameters and some don't, you can set the param values to nans in the correct shape to enable graph-batching.
    """
    def __init__(
        self,
        gradient_weight:float=10.0,
        energy_weight:float=1.0,
        param_weight:float=1e-3,
        weights:Dict[str,float]={"n2_k":1e-3, "n3_k":1e-2},
        skip_params_if_not_present:bool=True,
        ):

        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.gradient_weight = gradient_weight
        self.energy_weight = energy_weight
        self.param_weight = param_weight
        self.weights = weights
        self.skip_params_if_not_present = skip_params_if_not_present


    def forward(self, g):
        loss = 0
        graphs = dgl.unbatch(g)

        assert not (self.gradient_weight == 0 and self.energy_weight == 0 and self.param_weight == 0), "At least one of the weights must be non-zero."

        for graph in graphs:
            loss_term = 0

            if self.gradient_weight != 0:
                gradients = graph_utils.get_gradients(graph)
                gradients_ref = graph_utils.get_gradients(graph, suffix="_ref")
                loss_term = loss_term + self.mse_loss(gradients, gradients_ref) * self.gradient_weight

            if self.energy_weight != 0:
                energies = graph_utils.get_energies(graph)
                energies_ref = graph_utils.get_energies(graph, suffix="_ref")
                energy_contrib = self.mse_loss(energies, energies_ref) * self.energy_weight
                loss_term = loss_term + energy_contrib

            if self.param_weight != 0:
                try:
                    params_ref = graph_utils.get_parameters(graph, suffix="_ref")
                except KeyError as e:
                    if self.skip_params_if_not_present:
                        params_ref = {} # if this is empty, we will iterate over an empty list later, effectively skipping this part
                    else:
                        raise e

                params = graph_utils.get_parameters(graph)
                # concat all parameters
                param_tensor = []
                param_ref_tensor = []
                
                for k in params_ref.keys():
                    if 'improper' in k:
                        continue
                    fac = 1. if not k in self.weights.keys() else self.weights[k]
                    these_params = params[k]
                    these_params_ref = params_ref[k]
                    if k == 'n4_k':
                        # ensure that max_periodicity is that of the model, either by adding zeros or by removing columns
                        these_params_ref = correct_torsion_shape(these_params_ref, shape1=these_params.shape[1])

                    if these_params.shape != these_params_ref.shape:
                        raise ValueError(f"Shape of parameters {k} and {k}_ref do not match: {these_params.shape} vs {these_params_ref.shape}")

                    param_tensor.append(these_params.flatten()*fac)
                    param_ref_tensor.append(these_params_ref.flatten()*fac)

                if len(param_tensor) > 0:
                    param_tensor = torch.cat(param_tensor)
                    param_ref_tensor = torch.cat(param_ref_tensor)

                    diffs = torch.where(torch.isnan(param_tensor), torch.zeros_like(param_tensor), param_tensor - param_ref_tensor)

                    loss_contrib = torch.mean(torch.square(diffs))

                    loss_term = loss_term + loss_contrib * self.param_weight


            loss = loss + loss_term/len(graphs)
    
        return loss
    

def correct_torsion_shape(x, shape1):
        """
        Helper for bringing the torsion parameters into the correct shape. Adds zeros or cuts off the end if necessary.
        """
        if x.shape[1] < shape1:
            # concat shape1 - x.shape[1] zeros to the right
            return torch.cat([x, torch.zeros_like(x[:,:(shape1 - x.shape[1])])], dim=1)
        elif x.shape[1] > shape1:
            # w = Warning(f"n_periodicity ({shape1}) is smaller than the model torsion periodicity found ({x.shape[1]}).")
            # warnings.warn(w)
            return x[:,:shape1]
        else:
            return x
        



class ParameterLoss(torch.nn.Module):
    """
    Returns the MSE of the parameters of a graph asuming that they are stored at g.nodes[lvl].data[{k}/{eq}] and g.nodes[lvl].data[{k}/{eq}_ref]
    The SEs of the parameters are weighted by the square of the weight attribute. This is necessary because the units of the parameters are different.
    """
    def __init__(
        self, 
        weights:Dict[str,float]={"n2_k":1e-3, "n3_k":1e-2},
        param_names:List[str]=["n2_k", "n2_eq", "n3_k", "n3_eq", "n4_k"],
        average_params:bool=False,
        ):

        super().__init__()
        self.weights = weights
        self.mse_loss = torch.nn.MSELoss()
        self.param_names = param_names
        self.average_params = average_params


    def forward(self, g):
        loss = 0
        batch_size = g.num_nodes("g")
        params = graph_utils.get_parameters(g)
        params_ref = graph_utils.get_parameters(g, suffix="_ref")
        for param in self.param_names:
            if self.average_params:
                loss_term = self.mse_loss(params[param], params_ref[param])
            else:
                # divide by the number of batches, sum over the tuple dimension
                loss_term = torch.square(params[param] - params_ref[param]).sum() / batch_size

            if param in self.weights.keys():
                loss_term = loss_term * (self.weights[param]**2)

            loss = loss + loss_term
        
        return loss
    

class TuplewiseEnergyLoss(torch.nn.Module):
    """
    Assumes that the graph has been passed through grappa.models.Energy(suffix="_ref", write_suffix="_classical_ff", gradents=True)!

    For classical force fields, we can split the energy into contributions from specific interactions. Thus, we can use the individual energy contributions (before pooling!) as targets to obtain a more informative gradient signal.
    This is comparable to the more informative signal obained by matching gradients instead of energies but here, we not only have a spatial resolution but also a resolution in terms of the interaction term in the energy parameterization.
    In comparision to the ParameterLoss, this offers the following advantages:
        - The units of the loss contributions are the same for all parameters, no re-weighting is necessary.
        - We can learn classical contributions of improper torsion.
    """
    def __init__(
        self, 
        weights:Dict[str,float]={},
        interactions:List[str]=["n2", "n2", "n3", "n4", "n4_improper"],
        average_interactions:bool=False,
        ):

        super().__init__()
        self.weights = weights
        self.mse_loss = torch.nn.MSELoss()
        self.interactions = interactions
        self.average_interactions = average_interactions


    def forward(self, g):
        loss = 0
        batch_size = g.num_nodes("g")
        energies = graph_utils.get_tuplewise_energies(g)
        energies_ref = graph_utils.get_tuplewise_energies(g, suffix="_classical_ff")
        for interaction in self.interactions:
            if self.average_interactions:
                loss_term = self.mse_loss(energies[interaction], energies_ref[interaction])
            else:
                # divide by the number of batches, sum over the tuple dimension
                loss_term = torch.square(energies[interaction] - energies_ref[interaction]).sum() / batch_size
            if interaction in self.weights.keys():
                loss_term = loss_term * self.weights
            loss = loss + loss_term
        
        return loss
    

class EnergyLoss(torch.nn.Module):
    """
    Use energies as targets. The SEs of energies are averaged over conformation and batch.
    """
    def __init__(
        self, 
        energy_weight:float=1.0,
        ):

        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.energy_weight = energy_weight

    def forward(self, g):
        loss = 0
        energies = graph_utils.get_energies(g)
        energies_ref = graph_utils.get_energies(g, suffix="_ref")

        if self.energy_weight != 0:
            # mean over batch and conf
            loss = loss + self.mse_loss(energies, energies_ref) * self.energy_weight

        return loss
    

class GradientLoss(torch.nn.Module):
    """
    Use gradients as targets. The SEs of gradients are averaged over the spatial dimension, conformation divided by the batch_size and weighted by a hyperparameter, the gradient_weight. If average_forces is set to True, the SEs of the gradients are averaged over the atom dimension as well, which could be problematic because then the loss depends strongly on how the batch is being formed. (If the error is large for a force in a small molecule, this is surpressed if paired with a large molecule in a batch as opposed to another small molecule.)
    """
    def __init__(
        self, 
        gradient_weight:float=1.0,
        average_forces:bool=False
        ):

        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.gradient_weight = gradient_weight
        self.average_forces = average_forces

    def forward(self, g):
        loss = 0
        num_batches = g.num_nodes("g")
        gradients = graph_utils.get_gradients(g)
        gradients_ref = graph_utils.get_gradients(g, suffix="_ref")

        if self.gradient_weight != 0:
            squared_diffs = torch.square(gradients - gradients_ref)
            # shape of squared_diffs: (num_atoms_batch, num_confs, 3)
            if self.average_forces:
                # mean over all dimensions (the batch is already included implicity because the nodes are concatenated)
                loss = loss + torch.mean(squared_diffs) * self.gradient_weight
            else:
                # mean over conformation dimension and divide by number of batches (there is no batch dimensions, nodes are concatenated)
                loss = loss + torch.sum(torch.mean(squared_diffs, dim=1)) * self.gradient_weight / num_batches

        return loss
    

class GradientLossMolwise(torch.nn.Module):
    """
    Unbatch the graph and calculate the mse averaged over spatial dimension, conformation and atom. Then average these over the batch. Note: The averages do not commute because the number of atoms in the graph may differ. This way we assign the same weight to molecules with different amounts of atoms.
    """
    def __init__(
        self, 
        ):

        super().__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, g):
        loss = 0
        graphs = dgl.unbatch(g)

        for graph in graphs:
            gradients = graph_utils.get_gradients(graph)
            gradients_ref = graph_utils.get_gradients(graph, suffix="_ref")

            loss = loss + self.mse_loss(gradients, gradients_ref)/len(graphs)
    
        return loss
    