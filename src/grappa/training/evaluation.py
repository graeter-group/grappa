from grappa.utils.torch_utils import mean_absolute_error, root_mean_squared_error, invariant_mae, invariant_rmse
import numpy as np
from grappa.utils.graph_utils import get_energies, get_gradients, get_parameters
from grappa.utils import dgl_utils
from typing import List
import torch
from collections import defaultdict




class FastEvaluator:
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

        # re-batch the graphs: (no! breaks something)
        # g = dgl_utils.batch(graphs)

    def pool(self):
        """
        Returns a dictionary containing dictionaries for each dsname with the averaged metrics. For energies, this average is per conformation, for gradients it is per 3-vector, that is total number of atoms times conformations.
        Also resets the internal storage.
        The std_gradients entry is the atom-wise standard deviation, not the component-wise standard deviation.
        """
        metrics = {}

        for dsname in self.squared_error_energies.keys():
            metrics[dsname] = {}

            metrics[dsname]['rmse_energies'] = float(torch.sqrt(self.squared_error_energies[dsname].cpu() / self.num_energies[dsname]).detach().clone().item())
            metrics[dsname]['rmse_gradients'] = float(torch.sqrt(self.squared_error_gradients[dsname].cpu() / self.num_gradients[dsname]).detach().clone().item()) if self.gradients else None
            metrics[dsname]['crmse_gradients'] = float(torch.sqrt(self.squared_error_gradients[dsname].cpu() / self.num_gradients[dsname] / 3.).detach().clone().item()) if self.gradients else None


            if self.log_classical_values:
                metrics[dsname]['rmse_classical_gradients'] = float(torch.sqrt(self.squared_error_classical_gradients[dsname].cpu() / self.num_gradients[dsname]).detach().clone().item()) if self.gradients else None
                metrics[dsname]['rmse_classical_energies'] = float(torch.sqrt(self.squared_error_classical_energies[dsname].cpu() / self.num_energies[dsname]).detach().clone().item())


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

class Evaluator:
    """
    Does the same as the FastEvaluator but by unbatching the graphs and storing the energies and gradients explicitly on device RAM.
    """
    def __init__(self, keep_data=False, device='cpu', suffix='', suffix_ref='_ref', suffix_classical='_classical_ff', suffix_classical_ref:str=None, calculate_classical:bool=False):
        """
        keep_data: if True, the data is not deleted after pooling and can be used eg for plotting.
        """
        self.keep_data = keep_data
        self.device = device
        self.suffix = suffix
        self.suffix_ref = suffix_ref

        self.log_classical_values = calculate_classical

        self.suffix_classical = suffix_classical
        self.suffix_classical_ref = suffix_classical_ref


        self.init_storage()

    def init_storage(self):
        # initialize dictionaries that map dsname to a tensor holding the datapoints
        self.energies = {}
        self.gradients = {}

        self.reference_energies = {}
        self.reference_gradients = {}

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
                energies_classical = get_energies(g, suffix=self.suffix_classical).detach().flatten().to(self.device)

                if self.suffix_classical_ref is not None:
                    energies_ref_classical = get_energies(g, suffix=self.suffix_classical_ref).detach().flatten().to(self.device)

            # get the gradients in shape (n_atoms*n_confs, 3)
            gradients = get_gradients(g, suffix=self.suffix).detach().flatten(start_dim=0, end_dim=1).to(self.device)
            gradients_ref = get_gradients(g, suffix=self.suffix_ref).detach().flatten(start_dim=0, end_dim=1).to(self.device)
            if self.log_classical_values:
                gradients_classical = get_gradients(g, suffix=self.suffix_classical).detach().flatten(start_dim=0, end_dim=1).to(self.device)

                if self.suffix_classical_ref is not None:
                    gradients_ref_classical = get_gradients(g, suffix=self.suffix_classical_ref).detach().flatten(start_dim=0, end_dim=1).to(self.device)

            # store everything:
            self.energies.setdefault(dsname, []).append(energies)
            self.gradients.setdefault(dsname, []).append(gradients)

            self.reference_energies.setdefault(dsname, []).append(energies_ref)
            self.reference_gradients.setdefault(dsname, []).append(gradients_ref)


            if self.log_classical_values:
                self.classical_energies.setdefault(dsname, []).append(energies_classical)
                self.classical_gradients.setdefault(dsname, []).append(gradients_classical)
                
                if self.suffix_classical_ref is not None:
                    self.ref_classical_energies.setdefault(dsname, []).append(energies_ref_classical)
                    self.ref_classical_gradients.setdefault(dsname, []).append(gradients_ref_classical)


    def collect(self, bootstrap_seed=None):

        if bootstrap_seed is not None:
            # choose with replacement:
            np.random.seed(bootstrap_seed)
            mol_indices = {
                dsname:
                np.random.choice(len(self.energies[dsname]), size=len(self.energies[dsname]), replace=True).tolist()
                for dsname in self.energies.keys()
            }
        else:
            mol_indices = {dsname: range(len(self.energies[dsname])) for dsname in self.energies.keys()}

        self.n_mols = {}

        self.all_energies = {}
        self.all_gradients = {}

        self.all_reference_energies = {}
        self.all_reference_gradients = {}

        if self.log_classical_values:
            self.all_classical_energies = {}
            self.all_classical_gradients = {}

            if self.suffix_classical_ref is not None:
                self.all_ref_classical_energies = {}
                self.all_ref_classical_gradients = {}


        for dsname in self.energies.keys():

            self.n_mols[dsname] = len(self.energies[dsname])

            # concatenate the tensors
            self.all_energies[dsname] = torch.cat([self.energies[dsname][i] for i in mol_indices[dsname]], dim=0)
            self.all_gradients[dsname] = torch.cat([self.gradients[dsname][i] for i in mol_indices[dsname]], dim=0)

            self.all_reference_energies[dsname] = torch.cat([self.reference_energies[dsname][i] for i in mol_indices[dsname]], dim=0)
            self.all_reference_gradients[dsname] = torch.cat([self.reference_gradients[dsname][i] for i in mol_indices[dsname]], dim=0)

            if self.log_classical_values:
                self.all_classical_energies[dsname] = torch.cat([self.classical_energies[dsname][i] for i in mol_indices[dsname]], dim=0)
                self.all_classical_gradients[dsname] = torch.cat([self.classical_gradients[dsname][i] for i in mol_indices[dsname]], dim=0)

                if self.suffix_classical_ref is not None:
                    self.all_ref_classical_energies[dsname] = torch.cat([self.ref_classical_energies[dsname][i] for i in mol_indices[dsname]], dim=0)
                    self.all_ref_classical_gradients[dsname] = torch.cat([self.ref_classical_gradients[dsname][i] for i in mol_indices[dsname]], dim=0)


    def pool(self, n_bootstrap=0, seed=0):
        """
        If n_bootstrap == 0, the metrics are calculated once and returned.
        If n_bootstrap > 0, the metrics are calculated n_bootstrap times with different bootstrap samples and the mean and std of the metric values averaged over the bootstrap versions of the dataset are returned as mean = pool()[]
        """
        if n_bootstrap > 0:
            # first take the full dataset:
            self.collect()
            metrics = [self.get_metrics()]

            n_confs = {dsname: metrics[0][dsname]['n_confs'] for dsname in metrics[0].keys()}
            n_mols = {dsname: metrics[0][dsname]['n_mols'] for dsname in metrics[0].keys()}

            np.random.seed(seed)
            bootstrap_seeds = np.random.randint(0, 2**32, size=n_bootstrap-1).tolist()
            for i, bootstrap_seed in enumerate(bootstrap_seeds):
                self.collect(bootstrap_seed=bootstrap_seed)
                metrics.append(self.get_metrics())

            # now collect all values in a single dictionary, allowing that a dataset may not occur in one of the bootstrapped versions.
            all_metrics = {}

            all_ds_names = set()
            for i in range(len(metrics)):
                all_ds_names.update(metrics[i].keys())

            for dsname in all_ds_names:
                all_metrics[dsname] = [metrics[i][dsname] for i in range(len(metrics)) if dsname in metrics[i].keys()]

            # now calculate the mean and std for each metric
            for dsname in all_ds_names:
                all_metrics[dsname] = {key: {'mean': np.mean([m[key] for m in all_metrics[dsname]]), 'std': np.std([m[key] for m in all_metrics[dsname]])} for key in all_metrics[dsname][0].keys() if not key in ['n_confs', 'n_mols']}

                all_metrics[dsname]['n_confs'] = n_confs[dsname]
                all_metrics[dsname]['n_mols'] = n_mols[dsname]

            return all_metrics

        else:
            self.collect()
            return self.get_metrics()

    def get_metrics(self):

        metrics = {}
        for dsname in self.energies.keys():
            metrics[dsname] = {}

            metrics[dsname]['n_confs'] = self.all_energies[dsname].shape[0]

            metrics[dsname]['n_mols'] = self.n_mols[dsname]

            metrics[dsname]['std_energies'] = float(self.all_energies[dsname].std().detach().clone().item())
            metrics[dsname]['std_gradients'] = float(self.all_gradients[dsname].std().detach().clone().item() * np.sqrt(3))

            # calculate the metrics
            metrics[dsname]['rmse_energies'] = float(root_mean_squared_error(self.all_energies[dsname], self.all_reference_energies[dsname]).detach().clone().item())
            metrics[dsname]['mae_energies'] = float(mean_absolute_error(self.all_energies[dsname], self.all_reference_energies[dsname]).detach().clone().item())

            metrics[dsname]['rmse_gradients'] = float(invariant_rmse(self.all_gradients[dsname], self.all_reference_gradients[dsname]).detach().clone().item())
            metrics[dsname]['crmse_gradients'] = float(root_mean_squared_error(self.all_gradients[dsname], self.all_reference_gradients[dsname]).detach().clone().item())
            metrics[dsname]['mae_gradients'] = float(invariant_mae(self.all_gradients[dsname], self.all_reference_gradients[dsname]).detach().clone().item())

            if self.log_classical_values:

                if self.suffix_classical_ref is not None:
                    metrics[dsname]['rmse_classical_energies_from_ref'] = float(root_mean_squared_error(self.all_classical_energies[dsname], self.all_ref_classical_energies[dsname]).detach().clone().item())
                    metrics[dsname]['rmse_classical_gradients_from_ref'] = float(invariant_rmse(self.all_classical_gradients[dsname], self.all_ref_classical_gradients[dsname]).detach().clone().item())
                    metrics[dsname]['crmse_classical_gradients_from_ref'] = float(root_mean_squared_error(self.all_classical_gradients[dsname], self.all_ref_classical_gradients[dsname]).detach().clone().item())


        return metrics