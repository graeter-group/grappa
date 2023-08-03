"""
This class is neither written efficiently, nor well documented or tested. Only to be used for internal testing.
"""

from . import training
from ..models.energy import WriteEnergy
from . import utilities
from .utilities import get_param_statistics, get_grad, get_bonded_parameter_types

import dgl
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple, Union



class GrappaTrain(training.Train):
    """
    Training class with some methods configured to having a dataset of dgl graphs containing grappa data.
    This class is neither written efficiently, nor well documented or tested. Only to be used for internal testing.
    """
    # some additional parameters for automated evaluation
    def __init__(self, *args, reference_energy="u_ref", reference_forcefield="amber99_sbildn", energy_writer=WriteEnergy(),levels=["n2","n3"], energies=["bond", "angle", "total"], bond_help=0, angle_help=0, average=True, errors=False, by_atom=False, eval_interval=None, eval_forces=False, **kwargs):
        super(GrappaTrain, self).__init__(*args, **kwargs)
        self.reference_energy = reference_energy
        self.reference_forcefield = reference_forcefield
        self.levels = levels
        self.energies = energies
        self.bond_help = bond_help
        self.angle_help = angle_help
        self.average = average
        self.errors = errors
        self.by_atom  = by_atom
        self.energy_writer = energy_writer
        self.eval_interval = eval_interval
        self.eval_forces = eval_forces

    def training_epoch(self, model, do_all=False):
        if not self.eval_interval is None:
            if self.epoch%self.eval_interval == 0:
                print()
                print("evaluating...")
                model = model.cpu()
                if self.final_eval:
                    self.upon_end_of_training(model=model, plots=True, suffix="_"+str(self.epoch))
                model = model.to(self.device)
                print()
        return super().training_epoch(model, do_all)

    @staticmethod
    def get_targets_(model, batch, device=None, reference_energy="u_ref", energy="u", average=True, energy_writer=WriteEnergy()):
        if not device is None:
            batch = batch.to(device)
        batch = model(batch)
        batch = energy_writer(batch)
        if "weights" in list(batch.nodes["g"].data.keys()):
            raise RuntimeError("weighted case not implemented correctly with mean subtraction.")
            onehot_weights = out.nodes["g"].data["weights"]
            y_pred = out.nodes["g"].data[energy][onehot_weights!=0]
            y_true = out.nodes["g"].data[reference_energy][onehot_weights!=0]
        else:
            y_pred = batch.nodes["g"].data[energy]
            y_true = batch.nodes["g"].data[reference_energy]
            if average:
                y_pred = y_pred - y_pred.mean(dim=-1).unsqueeze(dim=-1)
                y_true = y_true - y_true.mean(dim=-1).unsqueeze(dim=-1)
        return y_pred, y_true
    

    

    def get_targets(self, model, batch, device=None):
        return GrappaTrain.get_targets_(model, batch, device, self.reference_energy, energy="u", average=self.average, energy_writer=self.energy_writer)

    def get_num_instances(self, batch):
        if "weights" in list(batch.nodes["g"].data.keys()):
            weights = batch.nodes["g"].data["weights"]
            return int(weights.sum())
        else:
            return batch.nodes["n1"].data["xyz"].shape[1]

    def get_collate_fn(self):
        def coll_fn(graphs):
            return dgl.batch(graphs)
        return coll_fn
    
    def init_metric_data(self):
        super(GrappaTrain, self).init_metric_data()
        if self.eval_forces:
            self.metric_data["f_mae_vl"] = []
            self.metric_data["f_rmse_vl"] = []
        self.metric_data["u_mae_vl"] = []
        self.metric_data["u_rmse_vl"] = []

    @staticmethod
    def get_forces(model, loader, device):
        ref = torch.cat([g.to(device).nodes["n1"].data["grad_ref"].flatten() for g in loader]).float()
        pred = torch.cat([utilities.get_grad(batch=g, model=model, energy_writer=WriteEnergy(), device=device, retain_graph=False)[0].flatten() for g in loader]).float()

        return pred, ref
    
    def evaluation(self, model):
        """
        overwrite the method.
        """
        # if self.epoch < self.direct_epochs:
        #     return super().evaluation(model)
        metric_values = {}

        energy_diffs = []
        if self.eval_forces:
            force_diffs = []

        for g in self.val_loaders[0]:
            
            # write the energy and forces into the graph
            ###############################
            if self.eval_forces:
                grads, model, g = utilities.get_grad(model=model, batch=g, device=self.device)

                with torch.no_grad():
                    force_diffs.append((g.nodes["n1"].data["grad_ref"] - grads).flatten().detach().clone().float())

            else:
                with torch.no_grad():
                    g = g.to(self.device)
                    g = model(g)
                    g = self.energy_writer(g)
            ###############################

            with torch.no_grad():
                e_ref = g.nodes["g"].data["u_ref"].detach().clone()
                e_ref -= e_ref.mean(dim=-1)

                e_pred = g.nodes["g"].data["u"].detach().clone()
                e_pred -= e_pred.mean(dim=-1)

                energy_diffs.append((e_ref-e_pred).flatten().float())

        with torch.no_grad():
            energy_diffs = torch.cat(energy_diffs).float()

            if self.eval_forces:
                force_diffs = torch.cat(force_diffs).float()


                key = "f_mae_vl"
                if key in self.metric_data.keys():
                    metric_values[key] = torch.nn.functional.l1_loss(force_diffs, torch.zeros_like(force_diffs)).detach().clone().to("cpu").item()
                key = "f_rmse_vl"
                if key in self.metric_data.keys():
                    metric_values[key] = torch.sqrt(torch.nn.functional.mse_loss(force_diffs, torch.zeros_like(force_diffs))).detach().clone().to("cpu").item()

            key = "u_mae_vl"
            if key in self.metric_data.keys():
                metric_values[key] = torch.nn.functional.l1_loss(energy_diffs, torch.zeros_like(energy_diffs)).detach().clone().to("cpu").item()

            key = "u_rmse_vl"
            if key in self.metric_data.keys():
                metric_values[key] = torch.sqrt(torch.nn.functional.mse_loss(energy_diffs, torch.zeros_like(energy_diffs))).detach().clone().to("cpu").item()

            assert len(self.current_train_loss) == 1
            avg_loss = None
            for tr_loader_key in self.current_train_loss.keys():
                try:
                    value = self.current_train_loss[tr_loader_key][0].detach().clone().cpu().item()
                except:
                    value = self.current_train_loss[tr_loader_key][0]
                num_inst = self.current_train_loss[tr_loader_key][1]
                avg_loss = value/num_inst

                metric_values["loss_tr"] = avg_loss


        for key in metric_values.keys():
            self.metric_data[key].append(metric_values[key])
        self.metric_data["epoch"].append(self.epoch)
        
        if len(self.metric_data[self.early_stopping_criterion]) > 0:
            val_loss = self.metric_data[self.early_stopping_criterion][-1]
        else:
            val_loss = float("inf")

        # update last and best state:
        if self.saving and (self.epoch%self.model_saving_interval == 0 and self.epoch != 0):

            # if loss is better than before, update the best state:
            if self.best_loss is None or self.best_loss > val_loss:
                with torch.no_grad():
                    torch.save(model.state_dict(), os.path.join(self.version_path, "best_model.pt"))
                    torch.save(self.optimizer.state_dict(), os.path.join(self.version_path, "best_opt.pt"))
                self.best_loss = val_loss

        model = model.to(self.device)
        model = model.train()

        return model


    def get_dataset_info(self, loader):
        stat_dic = {}
        num_atoms = []
        num_confs = []
        energies = []
        forces = []
        for batch in loader:
            for g in dgl.unbatch(batch):
                num_atoms.append(g.nodes["n1"].data["xyz"].shape[0])
                num_confs.append(g.nodes["n1"].data["xyz"].shape[1])
                energies.append(g.nodes["g"].data["u_ref"].flatten())
                if "grad_ref" in g.nodes["n1"].data.keys():
                    forces.append(g.nodes["n1"].data["grad_ref"].flatten())
        if len(forces) > 0:
            forces = torch.cat(forces)
            stat_dic["force var"] = forces.var().item()
            stat_dic["force l1"] = torch.nn.functional.l1_loss(forces, torch.zeros_like(forces)).item()

        energies = torch.cat(energies)
        stat_dic["energy var"] = energies.var().item()
        stat_dic["energy l1"] = torch.nn.functional.l1_loss(energies, torch.zeros_like(energies)).item()


        num_atoms = torch.tensor(num_atoms, dtype=torch.float32)
        num_confs = torch.tensor(num_confs, dtype=torch.float32)

        keys = ["atoms per molecule", "confs per molecule"]
        for i, arr in enumerate([num_atoms, num_confs]):
            stat_dic[keys[i]+" mean"] = arr.mean().item()
            stat_dic[keys[i]+" std"] = arr.std().item()
            stat_dic[keys[i]+" min"] = arr.min().item()
            stat_dic[keys[i]+" max"] = arr.max().item()
            if i==1:
                stat_dic["num_molecules"] = float(len(arr))

        try:
            if self.no_dataset_info:
                return stat_dic
        except AttributeError:
            pass

        path = os.path.join(self.version_path,"train_info")
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, "dataset_statistics")
        num_atoms = np.array(num_atoms)
        fig, ax = plt.subplots()
        plt.title("Number of atoms in molecules")
        ax.set_xlabel("num atoms")
        ax.set_ylabel("occurences")
        ax.hist(num_atoms, bins=20)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        fig.savefig(path+".png")
        return stat_dic

    @staticmethod
    def make_atomic_numbers(batch):
        h_nu = batch.nodes["n1"].data["h0"]
        atom_onehot = h_nu[:,:100]
        atoms = torch.argmax(atom_onehot, dim=1)
        batch.nodes["n1"].data["atoms"] = atoms
        n2_indices = batch.nodes["n2"].data["idxs"]
        batch.nodes["n2"].data["atoms"] = atoms[n2_indices]
        n3_indices = batch.nodes["n3"].data["idxs"]
        batch.nodes["n3"].data["atoms"] = atoms[n3_indices]
        n4_indices = batch.nodes["n4"].data["idxs"]
        batch.nodes["n4"].data["atoms"] = atoms[n4_indices]
        n4_improper_indices = batch.nodes["n4_improper"].data["idxs"]
        batch.nodes["n4_improper"].data["atoms"] = atoms[n4_improper_indices]
        return batch

    # for quantities that do not have a conf dimension
    @staticmethod
    def get_parameters(g, param_name="eq", level="n2", forcefield_name=""):
        if forcefield_name=="":
            key = param_name
        else:
            key = param_name+"_"+forcefield_name

        return g.nodes[level].data[key]


    @staticmethod
    def get_energy(g, energy_name="ref", level="g", forcefield_name=""):
        if forcefield_name=="":
            key = energy_name
        else:
            key = energy_name+"_"+forcefield_name

        if not "weights" in g.nodes["g"].data.keys():
            return g.nodes[level].data[key]
        else:
            assert False
            weights = g.nodes["g"].data["weights"]
            en = g.nodes[level].data[key]
            return en[weights!=0]


    @staticmethod
    def get_parameter_dict(model, loaders, param_names=["eq", "k"], levels=["n2", "n3"], forcefield_name="", dataset_names=None, model_device="cuda", out_device="cpu", energies=["bonded", "bond", "angle", "torsion", "ref", "reference_ff"], energy_factor=1., energy_writer=WriteEnergy(), atom_diff=False, grads=True):
        """
        Stores qm data in '_ref' entry.
        """

        assert not model_device is None
        model = model.to(model_device)

        if not isinstance(loaders, list):
            loaders = [loaders]
        
        do_grad = []
        grad_refs = []
        for loader in loaders:
            if len(loader) == 0:
                do_grad.append(False)
                grad_refs.append(False)
                continue
            g = next(iter(loader)).to(model_device)

            if f"grad_total_{forcefield_name}" in g.nodes["n1"].data.keys() and f"grad_nonbonded_{forcefield_name}" in g.nodes["n1"].data.keys() and grads:
                grad_refs.append(True)
            else:
                grad_refs.append(False)

            if "grad_ref" in g.nodes["n1"].data.keys() and grads:
                do_grad.append(True)
                grad, model, g = utilities.get_grad(model=model, batch=g, device=model_device)
            else:
                do_grad.append(False)
                g = model(g)
                g = energy_writer(g)

                
            for param_name in param_names:
                for level in levels:
                    if "n4" in level:
                        continue
                    if not forcefield_name =="":
                        if not param_name+"_"+forcefield_name in g.nodes[level].data.keys():
                            raise RuntimeError("A classical forcefield-parameter is not in the dataset: " + param_name+"_" +forcefield_name + " not in " + level)

            # for contrib in energies:
            #     # if forcefield_name !="" and contrib!="ref":
            #     #     if not "u_"+contrib+"_"+forcefield_name in out.nodes["g"].data.keys():
            #     #         raise RuntimeError("A classical forcefield-energy is not in the dataset: " + contrib)

        # start
        if dataset_names is None:
            if loaders is None:
                raise RuntimeError("specify the dataset names if not using a list of loaders.")
            dataset_names = [str(i) for i in range(len(loaders))]
            if len(loaders) == 1:
                dataset_names = [""]
        for name in dataset_names:
            if name!="" and name[0]!="_":
                name = "_"+name
            
        assert len(loaders) == len(dataset_names)
        out = {}
        for i, loader in enumerate(loaders):
            out[dataset_names[i]] = {}
        for i, loader in enumerate(loaders):
            # init dicts
            for level in levels:
                if not "n4" in level and atom_diff:
                    # init atom storage, store n atoms for the nth level
                    out[dataset_names[i]][level+"_atoms"] = torch.zeros((0,int(level[-1])), device=out_device)
                for param_name in param_names:
                    if "n4" in level:
                        param_name = "k"
                    out[dataset_names[i]][level + "_" +param_name+"_pred"] = torch.zeros(0, device=out_device)
                    if not forcefield_name =="":
                        out[dataset_names[i]][level + "_" +param_name+"_true"] = torch.zeros(0, device=out_device)
                    if "n4" in level:
                        break

            if do_grad[i]:
                out[dataset_names[i]]["grad_ref_pred"] = torch.zeros(0, device=out_device)
                out[dataset_names[i]]["grad_ref_true"] = torch.zeros(0, device=out_device)
                if "reference_ff" in energies:
                    out[dataset_names[i]]["grad_ref_ff_pred"] = torch.zeros(0, device=out_device)
                    out[dataset_names[i]]["grad_ref_ff_true"] = torch.zeros(0, device=out_device)


            for contrib in energies:
                en_name = "u_"+contrib
                out[dataset_names[i]][en_name+"_pred"] = torch.zeros(0, device=out_device)
                if forcefield_name!="" or contrib == "ref":
                    out[dataset_names[i]][en_name+"_true"] = torch.zeros(0, device=out_device)

            for batch in loader:
                batch = batch.to(model_device)
                if do_grad[i]:
                    grad, model, batch = utilities.get_grad(model=model, batch=batch, device=model_device)
                else:
                    batch = model(batch)
                    batch = energy_writer(batch)
                with torch.no_grad():
                    if atom_diff:
                        batch = GrappaTrain.make_atomic_numbers(batch)
                    for level in levels:
                        if not level in batch.ntypes:
                            continue
                        # for angle and bond, diff between atom types
                        if not "n4" in level and atom_diff:
                            atoms = GrappaTrain.get_parameters(batch, level=level, param_name="atoms", forcefield_name="")
                            out[dataset_names[i]][level+"_atoms"] = torch.cat((out[dataset_names[i]][level+"_atoms"], atoms.to(out_device)), dim=0)
                        ff_name = forcefield_name

                        for param_name in param_names:
                            if "n4" in level:
                                param_name = "k"
                            y_true = GrappaTrain.get_parameters(batch, level=level, param_name=param_name, forcefield_name=ff_name)
                            y_pred = GrappaTrain.get_parameters(batch, level=level, param_name=param_name, forcefield_name="")

                            out[dataset_names[i]][level + "_" +param_name+"_pred"] = torch.cat((out[dataset_names[i]][level + "_" +param_name+"_pred"], (y_pred.flatten()).to(out_device)))
                            if not forcefield_name =="":
                                out[dataset_names[i]][level + "_" +param_name+"_true"] = torch.cat((out[dataset_names[i]][level + "_" +param_name+"_true"], (y_true.flatten()).to(out_device)))
                            if "n4" in level:
                                break #only one param type: k

                    if do_grad[i]:
                        grad_true = batch.nodes["n1"].data[f"grad_qm"]
                        grad_pred = grad
                        out[dataset_names[i]]["grad_ref_pred"] = torch.cat((out[dataset_names[i]]["grad_ref_pred"], (grad_pred.flatten()).to(out_device)))
                        out[dataset_names[i]]["grad_ref_true"] = torch.cat((out[dataset_names[i]]["grad_ref_true"], (grad_true.flatten()).to(out_device)))

                        if "reference_ff" in energies:
                            grad_ff = batch.nodes["n1"].data[f"grad_total_{forcefield_name}"]

                            out[dataset_names[i]]["grad_ref_ff_pred"] = torch.cat((out[dataset_names[i]]["grad_ref_ff_pred"], (grad_ff.flatten()).to(out_device)))
                            out[dataset_names[i]]["grad_ref_ff_true"] = torch.cat((out[dataset_names[i]]["grad_ref_ff_true"], (grad_true.flatten()).to(out_device)))

                    for contrib in energies:
                        reference = False
                        if contrib == "bonded":
                            level = "u"
                        if contrib == "bonded_averaged":
                            level = "u"
                        elif contrib == "bond":
                            level = "u_n2"
                        elif contrib == "angle":
                            level = "u_n3"
                        elif contrib == "torsion":
                            level = "u_n4"
                        elif contrib == "improper":
                            level = "u_n4_improper"

                        en_name = "u_"+contrib
                        en_name_graph = en_name

                        if contrib=="total":
                            raise RuntimeError("total energy not implemented.")
                            en_name = "u_total"
                            y_true = GrappaTrain.get_energy(batch, level="g", energy_name="u_total", forcefield_name=forcefield_name)*energy_factor
                            y_pred = GrappaTrain.get_energy(batch, level="g", energy_name="u", forcefield_name="")*energy_factor + GrappaTrain.get_energy(batch, level="g", energy_name="u_nonbonded", forcefield_name=forcefield_name)*energy_factor
                            AVERAGE = True
                            if AVERAGE:
                                y_true = y_true - y_true.mean(dim=-1).unsqueeze(dim=-1)
                                y_pred = y_pred - y_pred.mean(dim=-1).unsqueeze(dim=-1)

                        elif contrib=="ref":
                            en_name = "u_ref"
                            y_true = GrappaTrain.get_energy(batch, level="g", energy_name="u", forcefield_name="qm")*energy_factor

                            y_pred = (GrappaTrain.get_energy(batch, level="g", energy_name="u", forcefield_name="")*energy_factor
                            +
                            GrappaTrain.get_energy(batch, level="g", energy_name="u_nonbonded", forcefield_name="ref")*energy_factor)

                            AVERAGE = True
                            if AVERAGE:
                                y_true = y_true - y_true.mean(dim=-1).unsqueeze(dim=-1)
                                y_pred = y_pred - y_pred.mean(dim=-1).unsqueeze(dim=-1)

                        elif contrib=="reference_ff":
                            en_name = "u_reference_ff"
                            y_true = GrappaTrain.get_energy(batch, level="g", energy_name="u", forcefield_name="qm")*energy_factor
                            y_pred = GrappaTrain.get_energy(batch, level="g", energy_name="u_total", forcefield_name=forcefield_name)*energy_factor
                            AVERAGE = True
                            if AVERAGE:
                                y_true -= y_true.mean(dim=-1).unsqueeze(dim=-1)
                                y_pred -= y_pred.mean(dim=-1).unsqueeze(dim=-1)

                        else:
                            y_pred = GrappaTrain.get_energy(batch, level="g", energy_name=level, forcefield_name="") * energy_factor
                            if not en_name_graph+"_"+forcefield_name in batch.nodes["g"].data.keys():
                                en_name_graph = level
                            if reference:
                                y_true = GrappaTrain.get_energy(batch, level="g", energy_name=en_name_graph, forcefield_name=forcefield_name) * energy_factor
                                y_true -= y_true.mean(dim=-1).unsqueeze(dim=-1)
                                y_pred -= y_pred.mean(dim=-1).unsqueeze(dim=-1)
                            elif contrib == "bonded_averaged":
                                y_pred = batch.nodes["g"].data["u"]*energy_factor
                                y_true = batch.nodes["g"].data["u_bonded_"+forcefield_name]*energy_factor
                                y_true -= y_true.mean(dim=-1).unsqueeze(dim=-1)
                                y_pred -= y_pred.mean(dim=-1).unsqueeze(dim=-1)
                            else:
                                y_true = GrappaTrain.get_energy(batch, level="g", energy_name=en_name_graph, forcefield_name=forcefield_name) * energy_factor

                        out[dataset_names[i]][en_name+"_pred"] = torch.cat((out[dataset_names[i]][en_name+"_pred"], (y_pred.flatten()).to(out_device)))
                        if forcefield_name !="" or contrib=="ref":
                            out[dataset_names[i]][en_name+"_true"] = torch.cat((out[dataset_names[i]][en_name+"_true"], (y_true.flatten()).to(out_device)))

        return out

    # scatterplot dividing for atoms involved (one scatterplot for each in partition_atoms and then the rest)
    @staticmethod
    def compare_atom_parameters(loaders=None, model=None, param_names=["eq", "k"], levels=["n2", "n3"], forcefield_name="", parameter_dict=None, dataset_names=None, min_y=None, max_y=None, show=False, bins_err=30, folder_path="atom_parameters", percentile=100, partition_atoms=[1,7,8], err_histogram=None, errors=False):
        if not isinstance(loaders, list):
            loaders = [loaders]
        if forcefield_name=="":
            return

        os.makedirs(folder_path, exist_ok=True)

        if dataset_names is None:
            if loaders is None:
                raise RuntimeError("specify the dataset names if not using a list of loaders.")
            dataset_names = [str(i) for i in range(len(loaders))]
            if len(loaders) == 1:
                dataset_names = [""]
        for name in dataset_names:
            if name!="" and name[0]!="_":
                name = "_"+name
 
        if parameter_dict is None:
            parameter_dict = GrappaTrain.get_parameter_dict(model=model,loaders=loaders, param_names=param_names, levels=levels, forcefield_name=forcefield_name, dataset_names=dataset_names)
        
        for key in parameter_dict.keys():
            for param_name in param_names:
                for level in levels:
                    if "n4" in level:
                        continue
                    y_true = parameter_dict[key][level + "_" +param_name+"_true"].detach().numpy()
                    y_pred = parameter_dict[key][level + "_" +param_name+"_pred"].detach().numpy()
                    atoms = parameter_dict[key][level+"_atoms"].detach().numpy()
                    
                    max_plotted = None
                    if min_y is None or max_y is None:
                        y_mean = np.mean(y_true)
                        max_plotted_a = np.percentile(np.abs(y_true-y_mean), percentile)
                        max_plotted_b = np.percentile(np.abs(y_pred-y_mean), percentile)
                        max_plotted = max(max_plotted_a, max_plotted_b)

                    if min_y is None:
                        min_y_ = y_mean - max_plotted
                    else:
                        min_y_ = min_y
                    if max_y is None:
                        max_y_ = y_mean + max_plotted
                    else:
                        max_y_ = max_y

                    fig, ax = plt.subplots(figsize=(8,8))
                    plt.title(level + " " + param_name)

                    id = np.array([min_y_, max_y_])
                    ax.plot(id,id, color="orange")
                    ax.set_xlim(min_y_, max_y_)
                    ax.set_ylim(min_y_, max_y_)

                    ax.set_xlabel(level + " " + param_name+"_true")
                    ax.set_ylabel(level + " " + param_name+"_pred")
                    already_covered = None
                    errs = []
                    labels = []
                    for a in partition_atoms:
                        atom_involved = (np.sum(np.where(atoms==a,1,0), axis=-1) > 0).astype(np.int64) # transform to int to execute logical operations

                        # exlcude those that were already covered
                        if already_covered is None:
                            already_covered = atom_involved
                        else:
                            atom_involved *= (1-already_covered) # only involve those that have not been covered yet.
                            already_covered += atom_involved # now only those atoms are involved that are not covered yet, so we can safely add

                        atom_involved = atom_involved==1 # transform back to bool
                        y_true_a = y_true[atom_involved]
                        y_pred_a = y_pred[atom_involved]

                        lab = "involving "+str(a)
                        if a != partition_atoms[0]:
                            lab += " and none of above"
                        ax.scatter(y_true_a, y_pred_a, label=lab, linewidths=0.1)
                        
                        errs.append(y_pred_a - y_true_a)
                        labels.append(lab)

                    ax.legend(fontsize="small")
                    fig.savefig(os.path.join(folder_path, level+"_"+param_name+key+"_by_atom.png"))
                    if show:
                        plt.show()
                    else:
                        plt.close(fig)
                        
                    plt.close("all")

                    if errors:
                        # now errors:
                        fig2, ax2 = plt.subplots(figsize=(8,8))
                        plt.title(level + " " + param_name + " errors")

                        err = err_histogram
                        all_errs = np.concatenate(errs).ravel()
                        err_mean = np.mean(all_errs)
                        if err is None:
                            err = np.percentile(np.abs(all_errs-err_mean), percentile)

                        ax2.hist(errs, label=labels, bins=bins_err, stacked=True, range=(err_mean-err, err_mean+err))
                        ax2.set_ylabel("occurences")
                        ax2.set_xlabel(param_name+"_pred - "+param_name+"_true")
                        ax2.legend()
                        fig2.savefig(os.path.join(folder_path, "err_"+level+"_"+param_name+key+"_by_atom.png"))
                        if show:
                            plt.show()
                        else:
                            plt.close(fig2)

        plt.close("all")



    # can be extended with metric values by providing a metric dict
    @staticmethod
    def compare_parameters(loaders=None, model=None, param_names=["eq", "k"], levels=["n2", "n3", "n4", "n4_improper"], forcefield_name="", parameter_dict=None, dataset_names=None, min_y=None, max_y=None, show=False, err_histogram=None, bins_err=30, bins_2d=100, folder_path="parameters", title_name=None, metric_dict=None, log_scale_accuracy=True, percentile=100, errors=False, verbose=False):
        if not isinstance(loaders, list):
            loaders = [loaders]
        if forcefield_name=="":
            return

        if dataset_names is None:
            if loaders is None:
                raise RuntimeError("specify the dataset names if not using a list of loaders.")
            dataset_names = [str(i) for i in range(len(loaders))]
            if len(loaders) == 1:
                dataset_names = [""]
        for name in dataset_names:
            if name!="" and name[0]!="_":
                name = "_"+name
 
        if parameter_dict is None:
            parameter_dict = GrappaTrain.get_parameter_dict(model=model,loaders=loaders, param_names=param_names, levels=levels, forcefield_name=forcefield_name, dataset_names=dataset_names, energies=[], atom_diff=False)

        for l_name in dataset_names:
            for level in levels:
                for param_name in param_names:
                    log_scale_accuracy_=log_scale_accuracy
                    if "n4" in level:
                        param_name = "k"
                        log_scale_accuracy_=True
                    try:
                        y_true = parameter_dict[l_name][level + "_" +param_name+"_true"]
                        y_pred = parameter_dict[l_name][level + "_" +param_name+"_pred"]
                    except ValueError:
                        if level in ["n3", "n4", "n4_improper"]:
                            continue
                        else:
                            raise

                    if len(y_pred)==0:
                        continue

                    metric_dict = training.Train.get_metrics_from_dict(parameter_dict, {"rmse": RMSE(), "mae": torch.nn.L1Loss()}, target_name=level + "_" +param_name)

                    ylabel = level+"_"+param_name+"_pred"
                    xlabel = level+"_"+param_name+"_" + forcefield_name

                    name = level+"_"+param_name+"_"+l_name
                    if verbose:
                        print(f"    Plotting {name} ...")

                    training.Train.visualize_targets(y_true=y_true, y_pred=y_pred, min_y=min_y, max_y=max_y, show=show, err_histogram=err_histogram, bins_err=bins_err, bins_2d=bins_2d, name=name, ylabel=ylabel, xlabel=xlabel, folder_path=folder_path, title_name=title_name, metric_dict=metric_dict, log_scale_accuracy=log_scale_accuracy_, percentile=percentile, loader_name=l_name, errors=errors)
                    plt.close("all")
                    # n4 has only one param name
                    if "n4" in level:
                        break

        return


    @staticmethod
    def compare_energies(model=None, loaders=None, dataset_names=None, show=False, device=None, min_y=None, max_y=None, err_histogram=None, bins_err=50, bins_2d=200, log_scale_accuracy=True,percentile=100, folder_path="energies",forcefield_name="gaff-1.81", energies=["bonded", "bond", "angle", "torsion", "improper", "ref", "reference_ff"], parameter_dict=None, errors=False, grads=True, verbose=False):
        assert forcefield_name!="" or energies==["ref"]

        if parameter_dict is None:
            parameter_dict = GrappaTrain.get_parameter_dict(model=model,loaders=loaders, param_names=[], levels=[], forcefield_name=forcefield_name, dataset_names=dataset_names, energies=energies, model_device=device, grads=grads, atom_diff=False)
        if dataset_names is None:
            if loaders is None:
                raise RuntimeError("specify the dataset names if not using a list of loaders.")
            dataset_names = [str(i) for i in range(len(loaders))]
            if len(loaders) == 1:
                dataset_names = [""]
        for name in dataset_names:
            if name!="" and name[0]!="_":
                name = "_"+name

        for l_name in dataset_names:

            contribs = energies.copy()
            if "grad_ref_true" in parameter_dict[l_name].keys() and "grad_ref_pred" in parameter_dict[l_name].keys() and grads:
                contribs += ["grad_ref"]
                if "grad_ref_ff_true" in parameter_dict[l_name].keys():
                    contribs += ["grad_ref_ff"]

            for contrib in contribs:
                if not "grad" in contrib:
                    en_name = "u_"+contrib
                else:
                    en_name = contrib

                y_true = parameter_dict[l_name][en_name+"_true"]
                y_pred = parameter_dict[l_name][en_name+"_pred"]

                if len(y_pred)==0:
                    continue
                try:
                    metric_dict = training.Train.get_metrics_from_dict(parameter_dict, {"rmse": RMSE(), "mae": torch.nn.L1Loss()}, target_name=en_name, lname=l_name)
                except:
                    if verbose:
                        print(f"    Error in {en_name} metric_dict, keeping it empty...")
                    

                ylabel = en_name+" pred"
                xlabel = en_name+" QM"
                if contrib == "reference_ff":
                    ylabel = "u "+forcefield_name
                    xlabel = "u QM"
                    en_name = f"u_{forcefield_name}"

                if contrib == "grad_ref_ff":
                    ylabel = "grad "+forcefield_name
                    xlabel = "grad QM"
                    en_name = f"grad_{forcefield_name}"

                if contrib == "grad_ref":
                    ylabel = "grad model"
                    xlabel = "grad QM"
                    en_name = f"grad_model"

                if contrib == "ref":
                    ylabel = "u model"
                    xlabel = "u QM"
                    en_name = f"u_model"


                fpath = os.path.join(folder_path, l_name)
                name = en_name

                if verbose:
                    print(f"    Plotting {name} ...")


                training.Train.visualize_targets(y_true=y_true, y_pred=y_pred, min_y=min_y, max_y=max_y, show=show, err_histogram=err_histogram, bins_err=bins_err, bins_2d=bins_2d, name=name, ylabel=ylabel, folder_path=fpath, title_name=en_name+" "+l_name+" in kcal/mol", metric_dict=metric_dict, loader_name=l_name, log_scale_accuracy=log_scale_accuracy, percentile=percentile, errors=errors, xlabel=xlabel)
                plt.close("all")
        return

        
    # for all params and energies, the graphs must contain the param+"_"+forcefield_name on all levels specified
    @staticmethod
    def compare_all(model, loaders, dataset_names=None, param_names=["eq", "k"], levels=["n2", "n3", "n4", "n4_improper"], energies=["bonded", "bond", "angle", "torsion", "improper", "ref", "reference_ff"], partition_atoms=[1,7,8], show=False, device=None, min_y=None, max_y=None, err_histogram=None, bins_err=30, bins_2d=100, log_scale_accuracy=False, percentile=100, folder_path="comparision", forcefield_name="gaff-1.81", parameter_dict=None, by_atom=False, errors=False, params=True, grads=True, verbose=False):
        
        model.eval()

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"


        if dataset_names is None:
            if loaders is None:
                raise RuntimeError("specify the dataset names if not using a list of loaders.")
            dataset_names = [str(i) for i in range(len(loaders))]
            if len(loaders) == 1:
                dataset_names = [""]
        for name in dataset_names:
            if name!="" and name[0]!="_":
                name = "_"+name

        os.makedirs(folder_path, exist_ok=True)

        if parameter_dict is None:
            if verbose:
                print("    Calculating parameters and energies for plots...")
            parameter_dict = GrappaTrain.get_parameter_dict(model=model,loaders=loaders, param_names=param_names, levels=levels, forcefield_name=forcefield_name, dataset_names=dataset_names, energies=energies, model_device=device, grads=grads, atom_diff=by_atom)

        GrappaTrain.compare_energies(min_y=min_y, max_y=max_y, show=show, err_histogram=err_histogram, bins_err=bins_err, bins_2d=bins_2d, folder_path=os.path.join(folder_path, "energies"), log_scale_accuracy=log_scale_accuracy, percentile=percentile, forcefield_name=forcefield_name, energies=energies, parameter_dict=parameter_dict, dataset_names=dataset_names, errors=errors, grads=grads, verbose=verbose)

        if params:
            GrappaTrain.compare_parameters(min_y=min_y, max_y=max_y, show=show, err_histogram=err_histogram, bins_err=bins_err, bins_2d=bins_2d, folder_path=os.path.join(folder_path, "parameters"), log_scale_accuracy=log_scale_accuracy, percentile=percentile, forcefield_name=forcefield_name, param_names=param_names, levels=levels, parameter_dict=parameter_dict, dataset_names=dataset_names, errors=errors, verbose=verbose)

        if by_atom:
            GrappaTrain.compare_atom_parameters(min_y=min_y, max_y=max_y, show=show, err_histogram=err_histogram, bins_err=bins_err, folder_path=os.path.join(folder_path, "atom_parameters"), percentile=percentile, forcefield_name=forcefield_name, param_names=param_names, levels=levels, parameter_dict=parameter_dict, partition_atoms=partition_atoms, dataset_names=dataset_names, errors=errors)


    def upon_end_of_training(self, model, plots=True, suffix=""):
        if not self.final_eval:
            return
        super().upon_end_of_training(plots=plots, model=torch.nn.Sequential(model, self.energy_writer), suffix=suffix)
        if not self.reference_forcefield is None and plots==True:
            print("Doing some parameter evaluation plots:")
            dset_names = self.dataset_names
            if dset_names is None:
                dset_names = [""]
            dnames = [n+"_vl" for n in dset_names]
            GrappaTrain.compare_all(model=torch.nn.Sequential(model, self.energy_writer), loaders=self.val_loaders, dataset_names=dnames, folder_path=os.path.join(self.version_path, "comparision"+suffix), forcefield_name=self.reference_forcefield, levels=self.levels, energies=self.energies, device="cpu")
        else:
            pass
            # dset_names = self.dataset_names
            # if dset_names is None:
            #     dset_names = [""]
            # dnames = [n+"_vl" for n in dset_names]
            # GrappaTrain.compare_energies(model=model, loaders=self.val_loaders, dataset_names=dnames, folder_path=os.path.join(self.version_path, "energies"), forcefield_name="ref", levels=[], energies=["ref"])

            print("Done")

    def additional_eval(self, model, path):
        if not self.reference_forcefield is None:
            dset_names = self.dataset_names
            if dset_names is None:
                dset_names = [""]
            dnames = [n+"_vl" for n in dset_names]
            GrappaTrain.compare_all(model=model, loaders=self.val_loaders, dataset_names=dnames, folder_path=os.path.join(path, "comparision"), forcefield_name=self.reference_forcefield, levels=self.levels, energies=self.energies)




class TrainSequentialParams(GrappaTrain):
    """
    Class for first training on parameters directly, then on energies and forces.
    This class is neither written efficiently, nor well documented or tested. Only to be used for internal testing. 
    """

    def __init__(self, *args, direct_epochs=10, bce_weight=0, classification_epochs=5, energy_factor=1., force_factor=0., param_factor=0., param_statistics=None, scale_dict={}, l2_dict={}, **kwargs):

        assert force_factor != 0 or energy_factor != 0 or param_factor != 0, "At least one of force_factor, energy_factor or param_factor must be non-zero."

        assert force_factor != 0 or energy_factor != 0 or direct_epochs > 0

        super().__init__(*args, **kwargs)

        if param_statistics is None:
            param_statistics = utilities.get_param_statistics

        self.param_statistics = param_statistics
        for key_ in self.param_statistics.keys():
            for key in self.param_statistics[key_].keys():
                self.param_statistics[key_][key] = self.param_statistics[key_][key].to(self.device)

        self.classification_epochs = classification_epochs
        self.direct_epochs = direct_epochs
        self.bce_weight = bce_weight
        self.force_factor = force_factor
        self.energy_factor = energy_factor
        self.param_factor = param_factor
        self.eval_forces = (force_factor != 0.)

        self.scale_dict = scale_dict
        self.l2_dict = l2_dict


    def get_loss(self, model, batch):
        loss = torch.tensor((0.), device=self.device).float()
        batch = batch.to(self.device)
        model = model.to(self.device)

        def grad_active():
            x = torch.tensor([1.], requires_grad=True)
            y = x * 2
            return y.requires_grad # is false if context is torch.no_grad()
    
        # recognize a possible context in which the function is called
        if self.target_for_memory:
            return torch.nn.functional.mse_loss(*self.get_en(batch, average=True))
        
        if self.force_factor != 0. and self.epoch >= self.direct_epochs:              
        
            if grad_active():
                pred, model, batch = utilities.get_grad(model, batch, self.device, self.energy_writer, retain_graph=True)
                ref = batch.nodes["n1"].data["grad_ref"]
                try:
                    loss = loss + torch.nn.functional.mse_loss(pred, ref) * self.force_factor
                except Exception as err:
                    if hasattr(err, 'message'):
                        err.message += f"\nshapes: {pred.shape}, {ref.shape}"
                        raise
            else:
                batch = model(batch)
                loss = float("nan")

        if self.energy_factor != 0. and self.epoch >= self.direct_epochs:
            if self.force_factor == 0. or not grad_active():
                batch = model(batch)
                batch = self.energy_writer(batch)
            pred, ref = self.get_en(batch, average=True)
            try:
                loss = loss + torch.nn.functional.mse_loss(pred, ref) * self.energy_factor
            except Exception as err:
                if hasattr(err, 'message'):
                    err.message += f"\nshapes: {pred.shape}, {ref.shape}"
                raise


        if self.epoch < self.direct_epochs:
            batch = model(batch)
            y_pred, y, y_l2 = self.get_scaled_params(batch, scale_dict=self.scale_dict, l2_dict=self.l2_dict)

            # use a huber loss of delta == 3 sigma for the parameters (since they are scaled by their standard deviation such that their std is 1)
            # Huber = torch.nn.HuberLoss(delta=3.)
            # loss += Huber(y_pred, y)

            if len(y_l2) > 0:
                loss += torch.nn.functional.mse_loss(y_l2, torch.zeros_like(y_l2))

            loss = loss + torch.nn.functional.mse_loss(y_pred, y)

        if self.param_factor != 0. or len(self.l2_dict) > 0:
            if self.energy_factor == 0 and not grad_active():
                batch = model(batch)
            
            if not "k" in batch.nodes["n2"].data.keys():
                batch = model(batch)

            y_pred, y, y_l2 = self.get_scaled_params(batch, scale_dict=self.scale_dict, l2_dict=self.l2_dict)

            # Huber = torch.nn.HuberLoss(delta=3.)
            # loss += Huber(y_pred, y) * self.param_factor
            try:
                if len(y_l2) > 0:
                    loss += torch.nn.functional.mse_loss(y_l2, torch.zeros_like(y_l2))

                if self.param_factor != 0.:
                    loss = loss + torch.nn.functional.mse_loss(y_pred, y) * self.param_factor
            except Exception as err:
                if hasattr(err, 'message'):
                    err.message += f"\nshapes: {pred.shape}, {ref.shape}"
                raise

        if self.epoch < self.classification_epochs:
            if self.bce_weight > 0:
                if self.energy_factor == 0 and not grad_active():
                    batch = model(batch)

                if "n4" in batch.ntypes:
                    loss = loss + self.bce_weight*torch.nn.functional.binary_cross_entropy_with_logits(batch.nodes["n4"].data["score"], batch.nodes["n4"].data["use_k"])
                if "n4_improper" in batch.ntypes:
                    loss = loss + self.bce_weight*torch.nn.functional.binary_cross_entropy_with_logits(batch.nodes["n4_improper"].data["score"], batch.nodes["n4_improper"].data["use_k"])

        return loss
    
    """
    Returns a list of tuples of level and parameter name.
    """
    @staticmethod
    def bonded_parameter_types():
        return [("n2", "k"), ("n2", "eq"), ("n3", "k"), ("n3", "eq"), ("n4", "k"), ("n4_improper", "k")]


    # def get_energy_loss(self, model, batch):
    #     if self.force_factor != 0:

    #     batch = batch.to(self.device)
    #     batch = model(batch)
    #     y_pred, y = self.get_targets_(batch)
    #     loss = torch.nn.functional.mse_loss(y_pred, y)
    #     return loss

    # returns the scaled and shifted parameters such that their std deviation is one
    def get_scaled_params(self, batch:dgl.DGLGraph, device="cpu", scale_dict:Dict={}, l2_dict:Dict={}):
        """
        Returns the ff parameters divided by the standard deviation of the classical parameter in the training set.
        scale_dict: scales the parameters that are in the kes of scale_dict by the value in the dict.
        l2_dict: applies an l2 loss to the parameters that are in the keys of l2_dict, where the parameters are scaled by the entry.
        """
        
        epsilon = 1e-2

        def get_scale_factor(level, param):
            x = 1./(epsilon+self.param_statistics["std"][level+"_"+param])
            x *= scale_dict[level+"_"+param] if scale_dict is not None and level+"_"+param in scale_dict.keys() else 1.
            return x


        FF = "ref"
        # transform the parameters such that they are normally distributed
        # we do not have do subtract the mean because this cancels out when taking the difference

        param_types = TrainSequentialParams.bonded_parameter_types()
        if not "n4_improper" in batch.ntypes:
            param_types = [p for p in param_types if p[0] != "n4_improper"]
        elif not "k" in batch.nodes["n4_improper"].data.keys():
            param_types = [p for p in param_types if p[0] != "n4_improper"]

        params_true = torch.cat(
            [
                (batch.nodes[level].data[param+"_"+FF].to(device)*get_scale_factor(level=level, param=param)).flatten()
                if level in batch.ntypes else torch.tensor([], device=device)
                for level, param in param_types
            ]
        , dim=0)

        try:
            params_pred = torch.cat(
                [
                    (batch.nodes[level].data[param].to(device)*get_scale_factor(level=level, param=param)).flatten()
                    if level in batch.ntypes else torch.tensor([], device=device)
                    for level, param in param_types
                ]
            , dim=0)
        except:
            for level in batch.ntypes:
                print(level, ":" , batch.nodes[level].data.keys())
            raise

        if l2_dict is None:
            params_l2 = torch.tensor([], device=device)
        elif len(list(l2_dict.keys()))==0:
            params_l2 = torch.tensor([], device=device)
        else:
            l2_dict_new = {}
            for k, v in l2_dict.items():
                if not "n4_improper" in k:
                    level, param = k.split("_")
                else:
                    level = "n4_improper"
                    param = k.split("_")[-1]
                if (level, param) in param_types:
                    l2_dict_new[(level, param)] = v
            params_l2 = torch.cat(
                [
                    (batch.nodes[level].data[param].to(device)*l2_dict_new[(level, param)]).flatten()
                    if level in batch.ntypes else torch.tensor([], device=device)
                    for level, param in l2_dict_new.keys()
                ]
            , dim=0)

                

        return params_pred, params_true, params_l2

    def get_en(self, batch, average=True):
        batch = self.energy_writer(batch)
        energies = batch.nodes["g"].data["u"]
        ref = batch.nodes["g"].data[self.reference_energy]

        if average:
            energies = energies - energies.mean(dim=-1).unsqueeze(dim=-1)
            ref = ref - ref.mean(dim=-1).unsqueeze(dim=-1)

        pred = energies.flatten()
        ref = ref.flatten()

        return pred, ref
    

    # fit params directly, then by energies and forces
    def get_targets(self, model, batch, device):
        batch = batch.to(device)
        batch = model(batch)
        return self.get_targets_(batch)
        
    # old version, is now handled in the get_loss, this is only used for the vl metrics
    # def get_targets_(self, batch, device="cpu"):
    #     if self.epoch >= self.direct_epochs or self.target_for_memory:
    #         return self.get_en(batch, average=self.average)
        
    #     else: # (self.epoch < self.direct_epochs)
    #         return self.get_scaled_params(batch, device=device)

    def get_targets_(self, batch, device="cpu"):
        return self.get_en(batch, average=self.average)
        
    def training_epoch(self, model, do_all=False):
        if self.epoch == self.direct_epochs:
            print()
            print("direct training finished, will continue on energies/forces")
            model = model.cpu()
            if self.final_eval:
                self.upon_end_of_training(model=model, plots=True, suffix="_direct")
            model = model.to(self.device)
        return super().training_epoch(model, do_all)
    
class RMSE(torch.nn.MSELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def forward(self, input, target):
        mse = super().forward(input, target)
        return torch.sqrt(mse)