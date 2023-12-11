from grappa.utils.torch_utils import mean_absolute_error, root_mean_squared_error, invariant_mae, invariant_rmse
import numpy as np
from grappa import utils
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
    def __init__(self, log_parameters=False, log_classical_values=False, metric_names:List[str]=None):
        """
        metric_names: list of strings for filtering metrics that should be logged. e.g. ['rmse_energies', 'rmse_gradients'] will only log these two metrics.
        """
        self.log_parameters = log_parameters
        self.log_classical_values = log_classical_values
        self.metric_names = metric_names

        self.init_storage()


    def init_storage(self):
        self.squared_error_energies = {}
        self.squared_error_gradients = {}

        self.abs_error_energies = {}
        self.abs_error_gradients = {}

        self.num_energies = {}
        self.num_gradients = {}

        if self.log_parameters:
            self.parameter_squarred_error = {}

        if self.log_classical_values:
            self.squared_error_classical_gradients = {}
            self.squared_error_classical_energies = {}


    def step(self, g, dsnames):

        # calculate the squared error and number of energies and gradients for each dataset
        ############################

        # get tensors of shape (n_batch) with squared errors and number of energies/gradients.
        # the number of gradients does not count the spatial dimension but the number of force vectors.
        energy_se, num_energies = utils.graph_utils.get_energy_se(g, l=2)
        gradient_se, num_gradients = utils.graph_utils.get_gradient_se(g, l=2)

        energy_ae, _ = utils.graph_utils.get_energy_se(g, l=1)
        gradient_ae, _ = utils.graph_utils.get_gradient_se(g, l=1)

        if not len(num_energies) == len(num_gradients) == len(dsnames):
            raise ValueError(f"Number of energies, gradients and dsnames must be equal but are {len(num_energies)}, {len(num_gradients)} and {len(dsnames)}")

        if not all(len(t.shape) == 1 for t in [energy_se, gradient_se, energy_ae, gradient_ae, num_energies, num_gradients]):
            raise ValueError(f"energy_se, gradient_se, energy_ae, gradient_ae, num_energies and num_gradients must be tensors of shape (n_batch) but are {[t.shape for t in [energy_se, gradient_se, energy_ae, gradient_ae, num_energies, num_gradients]]}")

        if self.log_classical_values:
            classical_energy_se, _ = utils.graph_utils.get_energy_se(g, suffix1='_classical_ff', suffix2='', l=2)
            classical_gradient_se, _ = utils.graph_utils.get_gradient_se(g, suffix1='_classical_ff', suffix2='', l=2)


        if self.log_parameters:
            parameter_se = utils.graph_utils.get_parameter_se(g, l=2)
            for ses, nums in parameter_se.values():
                if not len(ses.shape) == len(nums.shape) == 1:
                    raise ValueError(f"parameter squared errors and numbers must be tensors of shape (n_batch) but are {[t.shape for t in [ses, nums]]}")
                if not len(ses) == len(nums) == len(dsnames):
                    raise ValueError(f"Number of parameter squared errors and numbers must be equal but are {len(ses)}, {len(nums)} and {len(dsnames)}")


        # add the squared errors and number of energies and gradients to the storage
        ############################
        for i, dsname in enumerate(dsnames):
            if dsname not in self.squared_error_energies.keys():
                self.squared_error_energies[dsname] = 0
                self.squared_error_gradients[dsname] = 0
                self.abs_error_energies[dsname] = 0
                self.abs_error_gradients[dsname] = 0
                self.num_energies[dsname] = 0
                self.num_gradients[dsname] = 0
                if self.log_classical_values:
                    self.squared_error_classical_gradients[dsname] = 0
                    self.squared_error_classical_energies[dsname] = 0
            

            self.squared_error_energies[dsname] += energy_se[i].detach()
            self.squared_error_gradients[dsname] += gradient_se[i].detach()
            self.abs_error_energies[dsname] += energy_ae[i].detach()
            self.abs_error_gradients[dsname] += gradient_ae[i].detach()
            self.num_energies[dsname] += num_energies[i].detach()
            self.num_gradients[dsname] += num_gradients[i].detach()

            if self.log_classical_values:
                self.squared_error_classical_gradients[dsname] += classical_gradient_se[i].detach()
                self.squared_error_classical_energies[dsname] += classical_energy_se[i].detach()


            if self.log_parameters:
                if dsname not in self.parameter_squarred_error:
                    # initialize the dict but first detach all tensors
                    for ptype in parameter_se.keys():
                        parameter_se[ptype] = parameter_se[ptype][0].detach(), parameter_se[ptype][1].detach()

                    self.parameter_squarred_error[dsname] = {ptype: (ses[i], nums[i]) for ptype, (ses, nums) in parameter_se.items()}
                else:
                    for ptype in parameter_se.keys():
                        # add the se and number of parameters
                        self.parameter_squarred_error[dsname][ptype] = (
                            self.parameter_squarred_error[dsname][ptype][0] + parameter_se[ptype][0][i].detach(),
                            self.parameter_squarred_error[dsname][ptype][1] + parameter_se[ptype][1][i].detach()
                        )


    def pool(self):
        """
        Returns a dictionary containing dictionaries for each dsname with the averaged metrics. For energies, this average is per conformation, for gradients it is per 3-vector, that is total number of atoms times conformations.
        Also resets the internal storage.
        """
        metrics = {}

        for dsname in self.squared_error_energies.keys():
            metrics[dsname] = {}

            metrics[dsname]['rmse_energies'] = float(torch.sqrt(self.squared_error_energies[dsname].cpu() / self.num_energies[dsname].cpu()).item())
            metrics[dsname]['rmse_gradients'] = float(torch.sqrt(self.squared_error_gradients[dsname].cpu() / self.num_gradients[dsname].cpu()).item())
            metrics[dsname]['crmse_gradients'] = float(torch.sqrt(self.squared_error_gradients[dsname].cpu() / self.num_gradients[dsname].cpu() / 3.).item())

            metrics[dsname]['mae_energies'] = float((self.abs_error_energies[dsname].cpu() / self.num_energies[dsname].cpu()).item())
            metrics[dsname]['mae_gradients'] = float((self.abs_error_gradients[dsname].cpu() / self.num_gradients[dsname].cpu()).item())

            if self.log_classical_values:
                metrics[dsname]['rmse_classical_gradients'] = float(torch.sqrt(self.squared_error_classical_gradients[dsname].cpu() / self.num_gradients[dsname].cpu()).item())
                metrics[dsname]['rmse_classical_energies'] = float(torch.sqrt(self.squared_error_classical_energies[dsname].cpu() / self.num_energies[dsname].cpu()).item())

            if self.log_parameters:
                for ptype in self.parameter_squarred_error[dsname].keys():
                    metrics[dsname][f'rmse_{ptype}'] = float(torch.sqrt(self.parameter_squarred_error[dsname][ptype][0].cpu() / self.parameter_squarred_error[dsname][ptype][1].cpu()).item())


            # filter the metrics if necessary (lazy and inefficient)
            for key in metrics[dsname].keys():
                if not self.metric_names is None:
                    if key not in self.metric_names:
                        del metrics[dsname][key]

        # now calculate an averaged metric for the different datasets where each dataset gets the same weight, i.e. just form the average along the datasets:
        metrics['avg'] = {}
        for key in ['mae_energies', 'mae_gradients', 'rmse_energies', 'rmse_gradients']:
            if not self.metric_names is None:
                if key not in self.metric_names:
                    continue

            metrics['avg'][key] = np.mean([metrics[dsname][key] for dsname in metrics.keys() if dsname not in ['avg', 'all']])


        # reset the storage
        self.init_storage()

        return metrics

#%%

class ExplicitEvaluator:
    """
    Does the same as the Evaluator but by unbatching the graphs and storing the energies and gradients explicitly on device RAM.
    """
    def __init__(self, log_parameters=False, log_classical_values=False, keep_data=False, device='cpu', suffix='', suffix_ref='_ref', suffix_classical='_classical_ff'):
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

        self.init_storage()

    def init_storage(self):
        # initialize dictionaries that map dsname to a tensor holding the datapoints
        self.energies = {}
        self.gradients = {}

        self.reference_energies = {}
        self.reference_gradients = {}

        if self.log_parameters:
            self.parameters = {}
            self.reference_parameters = {}

        if self.log_classical_values:
            self.classical_energies = {}
            self.classical_gradients = {}


    def step(self, g, dsnames):
        # unbatch the graphs
        graphs = utils.dgl_utils.unbatch(g)

        if not len(graphs) == len(dsnames):
            raise ValueError(f"Number of graphs and dsnames must be equal but are {len(graphs)} and {len(dsnames)}")

        # get the energies and gradients
        for g, dsname in zip(graphs, dsnames):
            energies = utils.graph_utils.get_energies(g, suffix=self.suffix).detach().flatten().to(self.device)
            energies_ref = utils.graph_utils.get_energies(g, suffix=self.suffix_ref).detach().flatten().to(self.device)
            if self.log_classical_values:
                energies_classical = utils.graph_utils.get_energies(g, suffix=self.suffix_classical).flatten().to(self.device)

            # get the gradients in shape (n_atoms*n_confs, 3)
            gradients = utils.graph_utils.get_gradients(g, suffix=self.suffix).detach().flatten(start_dim=0, end_dim=1).to(self.device)
            gradients_ref = utils.graph_utils.get_gradients(g, suffix=self.suffix_ref).detach().flatten(start_dim=0, end_dim=1).to(self.device)
            if self.log_classical_values:
                gradients_classical = utils.graph_utils.get_gradients(g, suffix=self.suffix_classical).detach().flatten(start_dim=0, end_dim=1).to(self.device)

            if self.log_parameters:
                parameters = utils.graph_utils.get_parameters(g, exclude=[('n4_improper', 'k')], suffix=self.suffix)
                for p in parameters.values():
                    p = p.detach().flatten().to(self.device)

                ref_parameters = utils.graph_utils.get_parameters(g, suffix=self.suffix_ref, exclude=[('n4_improper', 'k')])
                for p in ref_parameters.values():
                    p = p.detach().flatten().to(self.device)

            # store everything:
            self.energies.setdefault(dsname, []).append(energies)
            self.gradients.setdefault(dsname, []).append(gradients)

            self.reference_energies.setdefault(dsname, []).append(energies_ref)
            self.reference_gradients.setdefault(dsname, []).append(gradients_ref)

            if self.log_classical_values:
                self.classical_energies.setdefault(dsname, []).append(energies_classical)
                self.classical_gradients.setdefault(dsname, []).append(gradients_classical)

            if self.log_parameters:
                for ptype, params in parameters.items():
                    self.parameters.setdefault(dsname, {}).setdefault(ptype, []).append(params)
                for ptype, params in ref_parameters.items():
                    self.reference_parameters.setdefault(dsname, {}).setdefault(ptype, []).append(params)


    def collect(self):
        for dsname in self.energies.keys():

            # concatenate the tensors
            self.energies[dsname] = torch.cat(self.energies[dsname], dim=0)
            self.gradients[dsname] = torch.cat(self.gradients[dsname], dim=0)

            self.reference_energies[dsname] = torch.cat(self.reference_energies[dsname], dim=0)
            self.reference_gradients[dsname] = torch.cat(self.reference_gradients[dsname], dim=0)

            if self.log_classical_values:
                self.classical_energies[dsname] = torch.cat(self.classical_energies[dsname], dim=0)
                self.classical_gradients[dsname] = torch.cat(self.classical_gradients[dsname], dim=0)

            if self.log_parameters:
                for ptype in self.parameters[dsname].keys():
                    self.parameters[dsname][ptype] = torch.cat(self.parameters[dsname][ptype], dim=0)
                    self.reference_parameters[dsname][ptype] = torch.cat(self.reference_parameters[dsname][ptype], dim=0)


    def pool(self):

        self.collect()

        metrics = {}
        for dsname in self.energies.keys():
            metrics[dsname] = {}

            metrics[dsname]['n_confs'] = self.energies[dsname].shape[0]

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

            if self.log_parameters:
                for ptype in self.parameters[dsname].keys():
                    metrics[dsname][f'rmse_{ptype}'] = float(root_mean_squared_error(self.parameters[dsname][ptype], self.reference_parameters[dsname][ptype]).item())


        if not self.keep_data:
            self.init_storage()

        return metrics