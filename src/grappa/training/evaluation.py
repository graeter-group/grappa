from grappa.utils.torch_utils import mean_absolute_error, root_mean_squared_error, invariant_mae, invariant_rmse
import numpy as np
from grappa.utils.graph_utils import get_energies, get_gradients, get_parameters
from grappa.utils import dgl_utils
from typing import List
import torch




class Evaluator:
    """
    Class that handles calculation of metrics for batched graphs and different datasets.
    Usage:
    for batch in loader:
        g, dsnames = batch
        g = model(g)
        evaluator.step(g, dsnames)
    metrics = evaluator.pool()

    the pool function returns a dictionary containing dictionaries for each dsname with the averaged metrics. For energies, this average is per conformation, for gradients it is per 3-vector, that is total number of atoms times conformations.
    """
    def __init__(self, log_parameters=False, log_classical_values=False, metric_names:List[str]=None, gradients:bool=True):
        """
        metric_names: list of strings for filtering metrics that should be logged. e.g. ['rmse_energies', 'rmse_gradients'] will only log these two metrics.
        """
        if log_parameters:
            raise NotImplementedError("Logging of parameters is not supported anymore.")
        
        self.log_classical_values = log_classical_values
        self.metric_names = metric_names

        self.gradients = gradients

        self.init_storage()


    def init_storage(self):
        self.squared_error_energies = {}
        self.squared_error_gradients = {}

        self.num_energies = {}
        self.num_gradients = {}

        if self.log_classical_values:
            self.squared_error_classical_gradients = {}
            self.squared_error_classical_energies = {}


    def step(self, g, dsnames):

        # calculate the squared error and number of energies and gradients for each dataset
        ############################

        graphs = dgl_utils.unbatch(g)
        for graph, dsname in zip(graphs, dsnames):
            
            energies_ref = get_energies(graph, suffix='_ref').detach()
            energies = get_energies(graph, suffix='').detach()
            assert len(energies.shape) == 1, f"energies must be a tensor of shape (n_confs) but is {energies.shape}"
            assert energies.shape[0] > 0, f"energies must be a tensor of shape (n_confs) but found n_conf = {energies.shape[0]}"
            assert energies.shape == energies_ref.shape, f"energies and energies_ref must have the same shape but are {energies.shape} and {energies_ref.shape}"
            energy_se = torch.sum(torch.square(energies - energies_ref))
            num_energies = energies.shape[0]

            if self.gradients:
                gradients = get_gradients(graph, suffix='').detach()
                gradients_ref = get_gradients(graph, suffix='_ref').detach()
                assert len(gradients.shape) == 3, f"gradients must be a tensor of shape (n_atoms,n_confs, 3) but is {gradients.shape}"
                assert gradients.shape[1] > 0, f"gradients must be a tensor of shape (n_atoms,n_confs, 3) but found n_conf = {gradients.shape[1]}"
                assert gradients.shape == gradients_ref.shape, f"gradients and gradients_ref must have the same shape but are {gradients.shape} and {gradients_ref.shape}"
                gradient_se = torch.sum(torch.square(gradients - gradients_ref))
                num_gradients = gradients.shape[0] * gradients.shape[1] # number of gradient vectors, not number of components


            if self.log_classical_values:
                classical_energies = get_energies(graph, suffix='_classical_ff').detach()
                classical_energy_se = torch.sum(torch.square(classical_energies - energies))

                if self.gradients:
                    classical_gradients = get_gradients(graph, suffix='_classical_ff').detach()
                    classical_gradient_se = torch.sum(torch.square(classical_gradients - gradients))     



            # add the squared errors and number of energies and gradients to the storage
            ############################
            if dsname not in self.squared_error_energies.keys():
                self.squared_error_energies[dsname] = 0
                self.squared_error_gradients[dsname] = 0
                self.num_energies[dsname] = 0
                self.num_gradients[dsname] = 0
                if self.log_classical_values:
                    self.squared_error_classical_gradients[dsname] = 0
                    self.squared_error_classical_energies[dsname] = 0
            

            self.squared_error_energies[dsname] += energy_se.detach()
            if self.gradients:
                self.squared_error_gradients[dsname] += gradient_se.detach()
            self.num_energies[dsname] += num_energies
            if self.gradients:
                self.num_gradients[dsname] += num_gradients

            if self.log_classical_values:
                self.squared_error_classical_gradients[dsname] += classical_gradient_se.detach()
                self.squared_error_classical_energies[dsname] += classical_energy_se.detach()


    def pool(self):
        """
        Returns a dictionary containing dictionaries for each dsname with the averaged metrics. For energies, this average is per conformation, for gradients it is per 3-vector, that is total number of atoms times conformations.
        Also resets the internal storage.
        """
        metrics = {}

        for dsname in self.squared_error_energies.keys():
            metrics[dsname] = {}

            metrics[dsname]['rmse_energies'] = float(torch.sqrt(self.squared_error_energies[dsname].cpu() / self.num_energies[dsname]).item())
            metrics[dsname]['rmse_gradients'] = float(torch.sqrt(self.squared_error_gradients[dsname].cpu() / self.num_gradients[dsname]).item()) if self.gradients else None
            metrics[dsname]['crmse_gradients'] = float(torch.sqrt(self.squared_error_gradients[dsname].cpu() / self.num_gradients[dsname] / 3.).item()) if self.gradients else None


            if self.log_classical_values:
                metrics[dsname]['rmse_classical_gradients'] = float(torch.sqrt(self.squared_error_classical_gradients[dsname].cpu() / self.num_gradients[dsname]).item()) if self.gradients else None
                metrics[dsname]['rmse_classical_energies'] = float(torch.sqrt(self.squared_error_classical_energies[dsname].cpu() / self.num_energies[dsname]).item())


            # filter the metrics if necessary (lazy and inefficient)
            for key in metrics[dsname].keys():
                if not self.metric_names is None:
                    if key not in self.metric_names:
                        del metrics[dsname][key]

        # now calculate an averaged metric for the different datasets where each dataset gets the same weight, i.e. just form the average along the datasets:
        metrics['avg'] = {}
        for key in ['rmse_energies', 'rmse_gradients']:
            if not self.metric_names is None:
                if key not in self.metric_names:
                    continue
            
            mlist = [metrics[dsname][key] for dsname in metrics.keys() if dsname not in ['avg', 'all'] and metrics[dsname][key] is not None]

            if len(mlist) == 0:
                metrics['avg'][key] = None
            else:
                metrics['avg'][key] = np.mean(mlist)


        # reset the storage
        self.init_storage()

        return metrics

#%%
# NOTE: ERROR IN RMSE_FROM_CLASSICAL. ONLY BONDED SHOULD BE USED THERE!!

class ExplicitEvaluator:
    """
    Does the same as the Evaluator but by unbatching the graphs and storing the energies and gradients explicitly on device RAM.
    """
    def __init__(self, log_parameters=False, log_classical_values=False, keep_data=False, device='cpu', suffix='', suffix_ref='_ref', suffix_classical='_classical_ff', ref_suffix_classical:str=None, molwise_rmse:bool=True):
        """
        keep_data: if True, the data is not deleted after pooling and can be used eg for plotting.
        """
        self.log_parameters = log_parameters
        self.log_classical_values = log_classical_values
        self.keep_data = keep_data
        self.device = device
        self.suffix = suffix
        self.suffix_ref = suffix_ref

        self.suffix_classical = suffix_classical

        self.ref_suffix_classical = ref_suffix_classical

        self.molwise_rmse = molwise_rmse

        self.init_storage()

    def init_storage(self):
        # initialize dictionaries that map dsname to a tensor holding the datapoints
        self.energies = {}
        self.gradients = {}

        self.reference_energies = {}
        self.reference_gradients = {}

        if self.molwise_rmse:
            self.energy_rmse_molwise = {}
            self.gradient_crmse_molwise = {}

        if self.log_parameters:
            self.parameters = {}
            self.reference_parameters = {}

        if self.log_classical_values:
            self.classical_energies = {}
            self.classical_gradients = {}

            self.ref_classical_energies = {}
            self.ref_classical_gradients = {}


    def step(self, g, dsnames):
        # unbatch the graphs
        graphs = dgl_utils.unbatch(g)

        if not len(graphs) == len(dsnames):
            raise ValueError(f"Number of graphs and dsnames must be equal but are {len(graphs)} and {len(dsnames)}")

        # get the energies and gradients
        for g, dsname in zip(graphs, dsnames):
            energies = get_energies(g, suffix=self.suffix).detach().flatten().to(self.device)
            energies_ref = get_energies(g, suffix=self.suffix_ref).detach().flatten().to(self.device)
            if self.log_classical_values:
                energies_classical = get_energies(g, suffix=self.suffix_classical).flatten().to(self.device)

                if self.ref_suffix_classical is not None:
                    energies_ref_classical = get_energies(g, suffix=self.ref_suffix_classical).flatten().to(self.device)

            # get the gradients in shape (n_atoms*n_confs, 3)
            gradients = get_gradients(g, suffix=self.suffix).detach().flatten(start_dim=0, end_dim=1).to(self.device)
            gradients_ref = get_gradients(g, suffix=self.suffix_ref).detach().flatten(start_dim=0, end_dim=1).to(self.device)
            if self.log_classical_values:
                gradients_classical = get_gradients(g, suffix=self.suffix_classical).detach().flatten(start_dim=0, end_dim=1).to(self.device)

                if self.ref_suffix_classical is not None:
                    gradients_ref_classical = get_gradients(g, suffix=self.ref_suffix_classical).detach().flatten(start_dim=0, end_dim=1).to(self.device)

            if self.log_parameters:
                parameters = get_parameters(g, exclude=[('n4_improper', 'k')], suffix=self.suffix)
                for p in parameters.values():
                    p = p.detach().flatten().to(self.device)

                ref_parameters = get_parameters(g, suffix=self.suffix_ref, exclude=[('n4_improper', 'k')])
                for p in ref_parameters.values():
                    p = p.detach().flatten().to(self.device)

            # store everything:
            self.energies.setdefault(dsname, []).append(energies)
            self.gradients.setdefault(dsname, []).append(gradients)

            self.reference_energies.setdefault(dsname, []).append(energies_ref)
            self.reference_gradients.setdefault(dsname, []).append(gradients_ref)

            if self.molwise_rmse:
                self.energy_rmse_molwise.setdefault(dsname, []).append(torch.sqrt(torch.mean(torch.square(energies - energies_ref))))
                self.gradient_crmse_molwise.setdefault(dsname, []).append(torch.sqrt(torch.mean(torch.square(gradients - gradients_ref))))

            if self.log_classical_values:
                self.classical_energies.setdefault(dsname, []).append(energies_classical)
                self.classical_gradients.setdefault(dsname, []).append(gradients_classical)
                
                if self.ref_suffix_classical is not None:
                    self.ref_classical_energies.setdefault(dsname, []).append(energies_ref_classical)
                    self.ref_classical_gradients.setdefault(dsname, []).append(gradients_ref_classical)

            if self.log_parameters:
                for ptype, params in parameters.items():
                    self.parameters.setdefault(dsname, {}).setdefault(ptype, []).append(params)
                for ptype, params in ref_parameters.items():
                    self.reference_parameters.setdefault(dsname, {}).setdefault(ptype, []).append(params)


    def collect(self):

        self.n_mols = {}

        for dsname in self.energies.keys():

            self.n_mols[dsname] = len(self.energies[dsname])

            # concatenate the tensors
            self.energies[dsname] = torch.cat(self.energies[dsname], dim=0)
            self.gradients[dsname] = torch.cat(self.gradients[dsname], dim=0)

            self.reference_energies[dsname] = torch.cat(self.reference_energies[dsname], dim=0)
            self.reference_gradients[dsname] = torch.cat(self.reference_gradients[dsname], dim=0)

            if self.molwise_rmse:
                self.energy_rmse_molwise[dsname] = torch.tensor(self.energy_rmse_molwise[dsname]).cpu().tolist()
                self.gradient_crmse_molwise[dsname] = torch.tensor(self.gradient_crmse_molwise[dsname]).cpu().tolist()

            if self.log_classical_values:
                self.classical_energies[dsname] = torch.cat(self.classical_energies[dsname], dim=0)
                self.classical_gradients[dsname] = torch.cat(self.classical_gradients[dsname], dim=0)

                if self.ref_suffix_classical is not None:
                    self.ref_classical_energies[dsname] = torch.cat(self.ref_classical_energies[dsname], dim=0)
                    self.ref_classical_gradients[dsname] = torch.cat(self.ref_classical_gradients[dsname], dim=0)

            if self.log_parameters:
                for ptype in self.parameters[dsname].keys():
                    self.parameters[dsname][ptype] = torch.cat(self.parameters[dsname][ptype], dim=0)
                    self.reference_parameters[dsname][ptype] = torch.cat(self.reference_parameters[dsname][ptype], dim=0)


    def pool(self):
        self.collect()
        return self.get_metrics()

    def get_metrics(self):

        metrics = {}
        for dsname in self.energies.keys():
            metrics[dsname] = {}

            metrics[dsname]['n_confs'] = self.energies[dsname].shape[0]

            metrics[dsname]['n_mols'] = self.n_mols[dsname]

            metrics[dsname]['std_energies'] = float(self.energies[dsname].std().item())
            metrics[dsname]['std_gradients'] = float(self.gradients[dsname].std().item() * np.sqrt(3))

            # calculate the metrics
            metrics[dsname]['rmse_energies'] = float(root_mean_squared_error(self.energies[dsname], self.reference_energies[dsname]).item())
            metrics[dsname]['mae_energies'] = float(mean_absolute_error(self.energies[dsname], self.reference_energies[dsname]).item())

            metrics[dsname]['rmse_gradients'] = float(invariant_rmse(self.gradients[dsname], self.reference_gradients[dsname]).item())
            metrics[dsname]['crmse_gradients'] = float(root_mean_squared_error(self.gradients[dsname], self.reference_gradients[dsname]).item())
            metrics[dsname]['mae_gradients'] = float(invariant_mae(self.gradients[dsname], self.reference_gradients[dsname]).item())

            if self.log_classical_values:
                metrics[dsname]['rmse_classical_energies'] = float(root_mean_squared_error(self.energies[dsname], self.classical_energies[dsname]).item())
                metrics[dsname]['rmse_classical_gradients'] = float(invariant_rmse(self.gradients[dsname], self.classical_gradients[dsname]).item())
                metrics[dsname]['crmse_classical_gradients'] = float(root_mean_squared_error(self.gradients[dsname], self.classical_gradients[dsname]).item())

                if self.ref_suffix_classical is not None:
                    metrics[dsname]['rmse_classical_energies_from_ref'] = float(root_mean_squared_error(self.classical_energies[dsname], self.ref_classical_energies[dsname]).item())
                    metrics[dsname]['rmse_classical_gradients_from_ref'] = float(invariant_rmse(self.classical_gradients[dsname], self.ref_classical_gradients[dsname]).item())
                    metrics[dsname]['crmse_classical_gradients_from_ref'] = float(root_mean_squared_error(self.classical_gradients[dsname], self.ref_classical_gradients[dsname]).item())

            if self.log_parameters:
                for ptype in self.parameters[dsname].keys():
                    metrics[dsname][f'rmse_{ptype}'] = float(root_mean_squared_error(self.parameters[dsname][ptype], self.reference_parameters[dsname][ptype]).item())


        if not self.keep_data:
            self.init_storage()

        return metrics