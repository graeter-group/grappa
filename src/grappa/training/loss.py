"""
A collection of loss functions for training grappa models.
"""

import dgl
import torch
from typing import Dict, List
from grappa.utils import graph_utils, dgl_utils


class MolwiseLoss(torch.nn.Module):
    """
    Unbatch the graph and calculate the mse averaged over spatial dimension, conformation and atom. Then average these over the batch. Note: The averages do not commute because the number of atoms in the graph may differ. This way we assign the same weight to molecules with different amounts of atoms.
    If skip_params_if_not_present is set to True, the parameters are only included in the loss if they are present in the graph. If a parameter is present but not a number, the loss contribution from these parameters is zero. Thus, if you have a dataset where some molecules have parameters and some don't, you can set the param values to nans in the correct shape to enable graph-batching.

    param_weights_by_dataset: Dictionary with keys corresponding to the dataset names and values corresponding to the weight of the parameter loss for this dataset. This overwrites the value of param_weight for entries of the datasets occuring in the dictionary.
    """
    def __init__(
        self,
        gradient_weight:float=0.8,
        energy_weight:float=1.0,
        param_weight:float=1e-3,
        tuplewise_weight:float=0,
        weights:Dict[str,float]={"n2_k":1e-3, "n3_k":1e-2, "n4_k":1e-4}, # only slightly change proper torsions when training on classical parameters.
        skip_params_if_not_present:bool=True,
        proper_regularisation:float=0., # prefactor for L2 regularisation of proper torsion parameters
        improper_regularisation:float=0.,        
        param_weights_by_dataset:Dict[str,float]={},
        terms:List[str]=['n2', 'n3', 'n4'],
        # min_k_angle:float=None,
        # min_k_bond:float=None,
        # penalty_k_angle:float=10.,
        # penalty_k_bond:float=10.,
        ):

        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.gradient_weight = gradient_weight
        self.energy_weight = energy_weight
        self.param_weight = param_weight
        self.tuplewise_weight = tuplewise_weight
        self.weights = weights
        self.skip_params_if_not_present = skip_params_if_not_present
        self.proper_regularisation = proper_regularisation
        self.improper_regularisation = improper_regularisation
        self.param_weights_by_dataset = param_weights_by_dataset
        self.terms = terms



    def forward(self, g, dsnames:List[str]=None):

        loss = 0
        graphs = dgl_utils.unbatch(g)

        assert not (self.gradient_weight == 0 and self.energy_weight == 0 and self.param_weight == 0), "At least one of the weights must be non-zero."

        for i, graph in enumerate(graphs):

            # loss per graph:
            loss_term = 0.

            if self.energy_weight != 0.:
                energies = graph_utils.get_energies(graph)
                energies_ref = graph_utils.get_energies(graph, suffix="_ref")
                assert energies.shape == energies_ref.shape, f"Shape of energies and energies_ref do not match: {energies.shape} vs {energies_ref.shape}"
                energy_contrib = self.mse_loss(energies, energies_ref) * self.energy_weight
                loss_term = loss_term + energy_contrib

            if self.gradient_weight != 0.:
                gradients = graph_utils.get_gradients(graph)
                gradients_ref = graph_utils.get_gradients(graph, suffix="_ref")
                assert gradients.shape == gradients_ref.shape, f"Shape of gradients and gradients_ref do not match: {gradients.shape} vs {gradients_ref.shape}"
                loss_term = loss_term + self.mse_loss(gradients, gradients_ref) * self.gradient_weight

            param_weight = self.param_weight

            if dsnames is not None:
                if dsnames[i] in self.param_weights_by_dataset.keys():
                    param_weight = self.param_weights_by_dataset[dsnames[i]]

            if param_weight != 0.:
                try:
                    params_ref = graph_utils.get_parameters(graph, suffix="_ref", terms=self.terms)
                except KeyError as e:
                    if self.skip_params_if_not_present:
                        params_ref = {} # if this is empty, we will iterate over an empty list later, effectively skipping this part
                    else:
                        raise e

                params = graph_utils.get_parameters(graph, terms=self.terms)
                # concat all parameters
                param_tensor = []
                param_ref_tensor = []
                
                for k in params_ref.keys():
                    if 'improper' in k:
                        # we do not train on improper parameters since we have three improper terms per interaction tuple
                        continue
                    fac = 1. if not k in self.weights.keys() else self.weights[k]
                    these_params = params[k]
                    these_params_ref = params_ref[k]

                    if k == 'n4_k':
                        # ensure that max_periodicity is that of the model, either by adding zeros or by removing columns
                        these_params_ref = correct_torsion_shape(these_params_ref, shape1=these_params.shape[1])

                    # set both parameters to zero where the ref params are nan. This enables batching graphs with params and without
                    these_params = torch.where(torch.isnan(these_params_ref), torch.zeros_like(these_params), these_params)
                    these_params_ref = torch.where(torch.isnan(these_params_ref), torch.zeros_like(these_params_ref), these_params_ref)


                    if these_params.shape != these_params_ref.shape:
                        raise ValueError(f"Shape of parameters {k} and {k}_ref do not match: {these_params.shape} vs {these_params_ref.shape}")

                    param_tensor.append(these_params.flatten()*fac)
                    param_ref_tensor.append(these_params_ref.flatten()*fac)

                if len(param_tensor) > 0:
                    param_tensor = torch.cat(param_tensor)
                    param_ref_tensor = torch.cat(param_ref_tensor)

                    diffs = param_tensor - param_ref_tensor

                    loss_contrib = torch.mean(torch.square(diffs))

                    loss_term = loss_term + loss_contrib * param_weight
    

            if self.proper_regularisation > 0.:
                propers = graph_utils.get_parameters(graph)['n4_k']
                if len(propers) > 0:
                    loss_term = loss_term + self.proper_regularisation * torch.mean(torch.square(propers))

            if self.improper_regularisation > 0.:
                impropers = graph_utils.get_parameters(graph)['n4_improper_k']
                if len(impropers) > 0:
                    loss_term = loss_term + self.improper_regularisation * torch.mean(torch.square(impropers))
                loss_term = loss_term + self.improper_regularisation * torch.mean(torch.square(impropers))

            # NOTE: this is currently not used
            assert self.tuplewise_weight == 0., f"Tuplewise loss not implemented yet, but weight is {self.tuplewise_weight}."
            if self.tuplewise_weight != 0.:
                tuplewise_energies = graph_utils.get_tuplewise_energies(graph)
                tuplewise_energies_ref = graph_utils.get_tuplewise_energies(graph, suffix="_classical_ff")

                pred_energies = []
                ref_energies = []

                for k, predicted_energies in tuplewise_energies.items():
                    if k not in tuplewise_energies_ref.keys():
                        raise ValueError(f"Tuplewise energy {k} not in tuplewise_energies_ref")
                    if tuplewise_energies[k].shape != tuplewise_energies_ref[k].shape:
                        raise ValueError(f"Shape of tuplewise energy {k} and {k}_ref do not match: {tuplewise_energies[k].shape} vs {tuplewise_energies_ref[k].shape}")

                    classical_energies = tuplewise_energies_ref[k]

                    # set all energies to zero where the ref energies are nan. This enables batching graphs with parameters and without
                    predicted_energies = torch.where(torch.isnan(classical_energies), torch.zeros_like(predicted_energies), predicted_energies)
                    classical_energies = torch.where(torch.isnan(classical_energies), torch.zeros_like(classical_energies), classical_energies)

                    pred_energies.append(predicted_energies.flatten())
                    ref_energies.append(classical_energies.flatten())

                pred_energies = torch.cat(pred_energies)
                ref_energies = torch.cat(ref_energies)
                
                tuplewise_loss_contrib = torch.mean(torch.square(pred_energies - ref_energies)) * self.tuplewise_weight

                loss_term = loss_term + tuplewise_loss_contrib

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
