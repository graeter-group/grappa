
import torch
import os
import copy
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpl_patches
import warnings
import time
import datetime
import shutil
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# change the weights stuff to using only a random set of batches from that loader
# hand over several ds, a partition and a seed directly, make class method that returns the splitted ds and the loaders.

# model_saving_interval: number of epochs after which the best and the last model are stored (overwrites the previous best and last model)
# checkpoint_interval: number of epochs after which a model is stored (and kept)\
# overwrite: determines whether a new version is instanciated with every run. if true, overwrites previous versions with the same name.
class Train:
    """
    This class is neither written efficiently, nor well documented or tested. Only to be used for internal testing.
    """

    # overwrite this method, must return a scalar to which .backward() can be applied for obtaining gradients
    def get_loss(self, model, batch):
        if self.loss_fn is None:
            raise RuntimeError("loss function must be specified or the method Trainer::get_loss must be overwritten")
        t = time.time()
        batch = batch.to(self.device)
        self.load_time += time.time() - t
        t = time.time()
        y_pred, y = self.get_targets(model, batch, self.device)
        self.forward_time += time.time() - t
        loss = self.loss_fn(y_pred, y)
        return loss

    def __init__(self, train_loader, val_loader, loss_fn=None, test_loader=None, evaluation_metrics={}, log_interval=1, print_interval=50, model_saving_interval=5, overwrite=False, figure_update_interval=50, batch_print_interval=20, storage_path=os.path.join("versions"), store_states=False, dataset_names=None, do_not_show=["loss_vl"], target_factor=1., clip_value=1e5, device="cpu"):

        assert not train_loader is None
        assert not val_loader is None

        self.target_for_memory = False
        self.device=device

        self.do_not_show = do_not_show
        self.target_factor = target_factor

        if not evaluation_metrics is None:
            self.evaluation_metrics = evaluation_metrics

        self.log_interval = log_interval
        self.print_interval = print_interval
        self.model_saving_interval = model_saving_interval
        self.overwrite = overwrite
        self.version_counter = 0
        self.figure_update_interval = figure_update_interval

        self.loss_fn = loss_fn

        if isinstance(train_loader, list):
            self.train_loaders = train_loader
        else:
            self.train_loaders = [train_loader]
        if isinstance(val_loader, list):
            self.val_loaders = val_loader
        else:
            self.val_loaders = [val_loader]
        if isinstance(test_loader, list):
            self.test_loaders = test_loader
        elif test_loader is None:
            self.test_loaders = None
        else:
            self.test_loaders = [test_loader]

        self.dataset_names = dataset_names
        if self.dataset_names is None and len(self.val_loaders)>1:
            self.dataset_names = [str(i) for i in range(len(self.val_loaders))]
        if not self.dataset_names is None:
            assert len(self.dataset_names) == len(self.val_loaders)

        self.batch_print_interval = batch_print_interval
        self.epoch = 0
        self.store_states = store_states

        self.evaluation_metrics = evaluation_metrics

        self.clip_value = clip_value

        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    ##################################################
    # CUSTOMIZE THESE BY INHERITENCE

    def get_optimizer(self, model, lr=1e-4):
        return torch.optim.Adam(lr=lr, params=model.parameters())

    def get_scheduler(self, optimizer):
        # meaning of threshold: to make the epoch count as better than before it has to reduce the loss by this amount, i.e. large threshold leads to early reduction in lr
        # in this setting "better" means lowering the train loss by one percent
        # step is called every log-interval on the total train loss

        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=1e-8)


    def get_warmup_scheduler(self, optimizer):
        # based on the pytorch_warmup library:
        # use the default values, i.e. the number of warmup steps is warmup_period = 2 / (1 - beta2)
        import pytorch_warmup as warmup
        return warmup.UntunedLinearWarmup(optimizer)
        


    # returns the predicted target and the true target depending on a batch as
    # y_pred, y_true
    # this method is to be changed upon inheritance
    def get_targets(self, model, batch, device=None, timed=False):
        x, y_true = batch
        if not device is None:
            x = x.to(device)
            y_true = y_true.to(device)
        y_pred = model(x)
        return y_pred, y_true

    def get_num_instances(self, batch):
        return batch[0].shape[0]

    def get_collate_fn(self):
        return None

    ##################################################

    # some parameters:
    # evaluation_metrics: dictionary mapping name-strings to callables that take two tensors as argument and return a scalar
    # more_metrics_on_train: if True, for every log interval, the evaluation_metrics are calculated for val and train set.
    # forced_optimizer: provides the option to manually hand over an optimizer that overwrites potential other optimizers that would be used instead, the learning rate is not affected by this
    # max lr from epoch: lambda function mapping epoch to lr, this can be reduced by the self.scheduler (thus "max" lr)
    # use self.scheduler can be a bool or a callable, returning a self.scheduler when called upon optimizer
    def run(self, model, epochs=100, lr=1e-4, loss_fn=None, version_name=None, device=None, mem_info=True, forced_optimizer=None, loss_show_factor=1., early_stopping_criterion=None, dataset_weights=None, min_lr_from_epoch=None, use_scheduler=False, ignore_warnings=True, eval_mode=True, log_note="", saving=True, final_eval=True, min_lr=None, show_times=False, use_warmup=False):

        self.final_eval = final_eval

        self.min_lr = min_lr
        self.show_times= show_times

        self.start_time = time.time()
        self.eval_mode=eval_mode
        self.saving = saving

        self.load_time = 0
        self.backward_time = 0
        self.forward_time = 0

        self.loss_fn = loss_fn
        self.loss_name = "loss"

        self.dataset_weights = dataset_weights
        if self.dataset_weights is None:
            self.dataset_weights = [1.]*len(self.train_loaders)
        self.min_lr_from_epoch = min_lr_from_epoch

        self.early_stopping_criterion = early_stopping_criterion
        if early_stopping_criterion is None:
            self.early_stopping_criterion = self.loss_name + "_vl"

        self.loss_show_factor = loss_show_factor

        if device is None:
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"
        else:
            self.device=device

        # evaluation and training metrics:
        self.metrics = {"loss": lambda y, y_pred: 0}
        self.metrics.update(self.evaluation_metrics)

        self.best_loss = float("inf")

        # determine version name:
        # NOTE: automate this
        self.version_name = version_name
        self.version_path = os.path.join(self.storage_path, str(self.version_name))
        os.makedirs(self.version_path, exist_ok=True)
        # handle recovery:
        
    
        self.clear()
            
        self.optimizer = self.get_optimizer(model, lr)
        self.epoch = 0


    
        # logging: # initializes self.metric_data
        self.init_metric_data()

        if not forced_optimizer is None:
            self.optimizer = forced_optimizer

        if use_warmup:
            self.warmup_scheduler = self.get_warmup_scheduler(self.optimizer)

        else:
            self.warmup_scheduler = None
        self.warmup_scheduler_helper = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=1.) # does nothing, only beacuse the warmup scheduler needs a scheduler to work


        if isinstance(use_scheduler, bool):
            if use_scheduler:
                self.scheduler = self.get_scheduler(self.optimizer)
            else:
                self.scheduler = None

        elif use_scheduler is None:
            self.scheduler = None

        else:
            self.scheduler = use_scheduler(self.optimizer)
        lr = self.optimizer.state_dict()["param_groups"][0]["lr"]


        self.params = {"lr":lr, "dataset_weights":self.dataset_weights, "epochs":epochs, "overwrite":False}
        self.old_params = {}
        if not self.min_lr_from_epoch is None:
            if self.min_lr_from_epoch(self.epoch) < self.params["lr"]:
                self.params["lr"] = self.min_lr_from_epoch(self.epoch)
        self.write_params()

        # keep track of the train loss:
        self.current_train_loss = {"tr": [0,0]}
        if not self.dataset_names is None:
            for l in self.dataset_names:
                name = "tr_" + l
                self.current_train_loss[name] = [0,0] # (total_loss, total_num_instances)

        # start the training:
        model = model.train()
        model = model.to(self.device)

        # log some parameters:
        self.log("#####################################")
        self.log("NEW RUN")
        self.log(log_note)
        info = {"loss": self.loss_name, "early_stop": self.early_stopping_criterion, "dataset_weights": "".join([str(w)+" " for w in self.dataset_weights])}
        df = pd.DataFrame(info, [""])
        self.log(str(df))
        self.log(" ")
        self.log("OPTIMIZER:")
        self.log(str(self.optimizer))
        self.log(" ")
        self.log("SCHEDULER:")
        if self.scheduler is None:
            self.log("NONE")
        else:
            self.log(str(self.scheduler.state_dict()))
        self.log(" ")
        print(str(df))

        num_params = count_parameters(model)

        self.log(" ")
        self.log("MODEL:")
        # sciencific notation:
        self.log(f"Number of parameters: {num_params:.2e}")
        self.log(" ")
        print(f"Number of parameters: {num_params:.2e}")

        # print some info on approx gpu memory
        if mem_info and "cuda" in str(self.device):
            self.memory_information(model)
        
        # print some info on the dataset:
        dataset_info = {}
        for i, l in enumerate(self.train_loaders):
            if not self.dataset_names is None:
                key = self.dataset_names[i]+"_tr"
            else:
                key = "tr"
            dataset_info[key] = self.get_dataset_info(l)

        for i, l in enumerate(self.val_loaders):
            if not self.dataset_names is None:
                key = self.dataset_names[i]+"_vl"
            else:
                key = "vl"
            dataset_info[key] = self.get_dataset_info(l)
    
        df = pd.DataFrame(dataset_info)
        pd.options.display.float_format = '{:.2E}'.format
        info = str("\n"+str(df)+"\n")
        self.log(info)
        print(info)

        
        print()
        print("starting the training:")
        print(self.get_print_header())
        self.log("TIME: " + str(datetime.datetime.now()))
        self.log("#####################################")
        self.log(self.get_log_header())

        with warnings.catch_warnings():
            if ignore_warnings:
                warnings.simplefilter("ignore")
            
            self.more_training = True

            try:
                self.epochs = epochs
                while self.epoch < self.epochs and self.more_training:
                    if self.epoch == epochs-1 or not self.more_training:
                        model = self.training_epoch(model, do_all=True)
                    else:
                        model = self.training_epoch(model, do_all=False)
                    self.epoch += 1

                with torch.no_grad():
                    self.log(f"storing last model... at {os.path.join(self.version_path, 'last_model.pt')}")
                    torch.save(model.state_dict(), os.path.join(self.version_path, "last_model.pt"))
                if self.final_eval:
                    self.upon_end_of_training(model=model)
                return model


            except KeyboardInterrupt:
                with torch.no_grad():
                    self.log(f"storing last model... at {os.path.join(self.version_path, 'last_model.pt')}")
                    torch.save(model.state_dict(), os.path.join(self.version_path, "last_model.pt"))
                # model = model.cpu()
                if self.final_eval:
                    self.upon_end_of_training(model=model)
                return model



    # suffix does nothing, only for inheritance
    def upon_end_of_training(self, model, plots=True, suffix=""):
        if "cuda" in self.device:
            torch.cuda.empty_cache()
        print()
        print("Evaluating. May take some time.")
        print("You can savely interrupt this process.")
        self.log("")
        if self.eval_mode:
            test_data = self.get_test_metrics(model=model.eval().to(self.device), test_loaders=[], test_dataset_names=[])
        else:
            test_data = self.get_test_metrics(model=model.to(self.device), test_loaders=[], test_dataset_names=[])
        df = pd.DataFrame(test_data)
        pd.options.display.float_format = '{:.2E}'.format
        self.log(str(df))
        print(df)
        print()
        if plots:
            print("Creating evaluation plots...")
            plots = self.evaluate_model(model=model, path=self.version_path)
            # with open(os.path.join(self.version_path, "evaluation", "plot_container.pickle"), "bw") as f:
            #     pickle.dump(plots, f)
            print("Done")

    def additional_eval(self, model, path):
        pass

    def training_epoch(self, model, do_all=False):
        try:
            self.read_params()
        except json.decoder.JSONDecodeError:
            pass

        if self.params["do_eval"] == 1:
            self.params["do_eval"] = 0
            self.evaluate_model(model=model, path=self.version_path)
            self.additional_eval(model=model, path=self.version_path)
            self.old_params = None # force the overwriting
            self.write_params()
            
        self.epochs = self.params["epochs"]
        self.dataset_weights = self.params["dataset_weights"]
        lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        if lr != self.params["lr"]:
            if self.params["overwrite"]:
                opt_dict = self.optimizer.state_dict()
                opt_dict["param_groups"][0]["lr"] = self.params["lr"]
                self.optimizer.load_state_dict(opt_dict)
                if not self.scheduler is None:
                    # reset the self.scheduler after lr has been set by hand and trigger the cooldown
                    self.scheduler._reset()
                    self.scheduler.cooldown_counter = 0


        # used to compute running average of train loss:
        for key in self.current_train_loss.keys():
            self.current_train_loss[key] = [0,0]

        weights = np.array(self.dataset_weights, dtype=np.float32)
        assert np.max(weights > 0)
        weights /= np.max(weights)

        idx = 0
        num_batches = sum([len(l) for l in self.train_loaders])
        for i_loader, tr_loader in enumerate(self.train_loaders):
            for i_batch, batch in enumerate(tr_loader):
                # drop out of the data loader in this epoch, with probability according to the dataset weights
                if weights[i_loader] < 1.:
                    skip = np.random.choice([True, False], size=1, p=[1.-weights[i_loader], weights[i_loader]])[0]
                    if skip:
                        continue

                self.optimizer.zero_grad()

                loss = self.get_loss(model=model, batch=batch)

                b = time.time()
                loss.backward()
                self.backward_time += time.time() - b

                torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)

                self.optimizer.step()

                # warmup, see approach 2 in https://github.com/Tony-Y/pytorch_warmup:
                if not self.warmup_scheduler is None:
                    with self.warmup_scheduler.dampening():
                        self.warmup_scheduler_helper.step()

                with torch.no_grad():
                    # assume that the loss averages over instances. calculate the average over the whole train set, not just the batch
                    # (different from averaging over batches when the number of instances per batch varies):
                    num_instances = self.get_num_instances(batch)
                    if not self.dataset_names is None:
                        # loader-specific:
                        tr_dict_key = "tr_" + self.dataset_names[i_loader]
                        self.current_train_loss[tr_dict_key][0] += loss * num_instances * self.loss_show_factor
                        self.current_train_loss[tr_dict_key][1] += num_instances
                        # all:
                    self.current_train_loss["tr"][0] += loss * num_instances * self.loss_show_factor
                    self.current_train_loss["tr"][1] += num_instances

                if (idx%self.batch_print_interval == 0 and idx!=0) or idx == num_batches-1:
                    self.print_batch_update(idx)
                
                idx+=1

        
        if self.epoch%self.log_interval == 0 or do_all:
            model = self.evaluation(model)
            self.log()
            if not self.scheduler is None:
                # commented out due for independency of pytorch-warmup
                # with self.warmup_scheduler.dampening():
                #     # self.scheduler criterion is the last train loss:
                #     self.scheduler.step(self.metric_data[self.loss_name+"_tr"][-1])
                try:
                    self.scheduler.step(self.metric_data[self.loss_name+"_tr"][-1])
                except IndexError:
                    pass
                 
        # write the new lr to the param file
        lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        if lr != self.params["lr"]:
            self.params["lr"] = self.optimizer.state_dict()["param_groups"][0]["lr"]

        if self.epoch%(self.log_interval*40) == 0 and self.epoch!=1:
            self.log(self.get_log_header())

        if self.epoch%self.model_saving_interval == 0 or do_all:
            if not self.epoch%self.log_interval == 0:
                model = self.evaluation(model)

        if not self.batch_print_interval is None:
            self.print_batch_update(idx)
        if self.epoch%self.print_interval == 0 or do_all:
            # self.print_batch_update(idx)
            print()

        if not self.figure_update_interval is None:
            if (self.epoch%self.figure_update_interval == 0 and self.epoch != 0) or do_all:
                try:
                    self._make_loss_figure()
                except:
                    pass

        if not self.min_lr_from_epoch is None:
            if self.min_lr_from_epoch(self.epoch) < self.params["lr"]:
                self.params["lr"] = self.min_lr_from_epoch(self.epoch)
        self.write_params()
        if not self.min_lr is None:  
            if self.params["lr"] < self.min_lr:
                self.more_training = False
                self.log("stopped training because lr is below min_lr")

        return model
    

    # evaluate the model on the whole dataset.
    # append data to the metric_data dictionary
    def evaluation(self, model):
        if self.eval_mode:
            model = model.eval()

        metric_values = self.get_metrics(model, metrics=self.metrics, during_train=True)

        # update the metric_data dictionary by reading the metric value nested dictionaries:
        for l_name in metric_values.keys(): # loader name
            for m_name in metric_values[l_name].keys():
                key = m_name + "_" + l_name
                self.metric_data[key].append(metric_values[l_name][m_name])
        self.metric_data["epoch"].append(self.epoch)
        
        val_loss = self.metric_data[self.early_stopping_criterion][-1]

        # update last and best state:
        if self.saving and (self.epoch%self.model_saving_interval == 0 and self.epoch != 0):
            with torch.no_grad():
                # currently, save only the best model, not the last
                pass
                # torch.save(model.state_dict(), os.path.join(self.version_path, "last_model.pt"))
                # torch.save(self.optimizer.state_dict(), os.path.join(self.version_path, "last_opt.pt"))


            # if loss is better than before, update the best state:
            if self.best_loss is None or self.best_loss > val_loss:
                with torch.no_grad():
                    torch.save(model.state_dict(), os.path.join(self.version_path, "best_model.pt"))
                    torch.save(self.optimizer.state_dict(), os.path.join(self.version_path, "best_opt.pt"))
                self.best_loss = val_loss


        model = model.to(self.device)
        model = model.train()
        return model

    # returns a dictionary containing the total values (summed over all instances) for each metric
    @staticmethod
    def get_total_metrics(model, metrics:dict, loader, get_targets, get_num_instances, metrics_are_averaged=True, suffix="", device="cpu", target_factor=1., get_loss=None):
        out = {}
        total_num_instances = 0
        for mname in metrics.keys():
            out[mname+suffix] = 0
            for batch in loader:
                y_pred, y = (v.detach().clone() for v in get_targets(model=model, batch=batch, device=device))
                y_pred *= target_factor
                y *= target_factor

                num_instances = get_num_instances(batch)
                total_num_instances += num_instances

                if mname == "loss":
                    if get_loss is None:
                        continue
                    else:
                        mfun = get_loss
                        
                mfun = metrics[mname]
                batchloss = mfun(y_pred, y)
                if metrics_are_averaged:
                    out[mname+suffix] += batchloss * num_instances
                else:
                    out[mname+suffix] += batchloss

        for mname in metrics.keys():
            try:
                if isinstance(out[mname+suffix], torch.Tensor):
                    out[mname+suffix] = out[mname+suffix].item()
            except:
                if math.isnan(out[mname+suffix]):
                    out[mname+suffix] = float("nan")
                    continue
                else:
                    raise
                
        return out, total_num_instances
        

    # returns a dictionary of metric_names containing a dictionary of dataset_names mapped to metric values
    # if average is True, the metric values will be averaged, else, every dict entry will be a tuple of total value and total number of instances
    @staticmethod
    def calc_metrics(model, metrics:dict, loaders, dataset_names, get_targets, get_num_instances, metrics_are_averaged=True, average=True, device="cpu", target_factor=1., loss_factor=1., get_loss=None):
        assert loss_factor == 1., "this is deprecated"
        assert target_factor == 1., "this is deprecated"

        out = {}
        for idx, loader in enumerate(loaders):
            l_name = dataset_names[idx]
            l_out, t_num_inst = Train.get_total_metrics(model=model, metrics=metrics, loader=loader, metrics_are_averaged=metrics_are_averaged, device=device, get_targets=get_targets, get_num_instances=get_num_instances, target_factor=target_factor, get_loss=get_loss)
            out[l_name] = l_out
            for key in l_out.keys():
                if average:
                    out[l_name][key] = out[l_name][key] / t_num_inst
                else:
                    out[l_name][key] = [out[l_name][key], t_num_inst]
        
        return out

    def get_metrics(self, model, metrics=None, additional_loaders=[], add_dataset_names=[], during_train=False, average=True, target_factor=None):
        if target_factor is None:
            target_factor = self.target_factor
        assert len(additional_loaders) == len(add_dataset_names)
        if metrics is None:
            metrics=self.metrics
        loaders = additional_loaders
        dataset_names_ = add_dataset_names
        loaders += self.val_loaders
        if self.dataset_names is None: # if there is only one loader
            val_names = ["dummy"]
        else:
            val_names = ["vl_" + n for n in self.dataset_names]
        dataset_names_ += val_names

        out = Train.calc_metrics(model=model, metrics=metrics, loaders=loaders, dataset_names=dataset_names_, average=False, get_targets=self.get_targets, get_num_instances=self.get_num_instances, device=self.device, target_factor=target_factor, loss_factor=self.loss_show_factor, get_loss=self.get_loss)
        out["vl"] = {}
        for mname in metrics.keys():
            total_loss = sum([out[l_name][mname][0] for l_name in val_names])
            total_inst = sum([out[l_name][mname][1] for l_name in val_names])
            out["vl"][mname] = [total_loss, total_inst]
        if self.dataset_names is None:
            out.pop("dummy") # if there is only one loader


        if not during_train:
            # recalculate
            if self.dataset_names is None: # if there is only one loader
                tr_names = ["dummy"]
            else:
                tr_names = ["tr_" + n for n in self.dataset_names]
            out_tr = Train.calc_metrics(model=model, metrics=metrics, loaders=self.train_loaders, dataset_names=tr_names, average=False, get_targets=self.get_targets, get_num_instances=self.get_num_instances, device=self.device, target_factor=target_factor, loss_factor=self.loss_show_factor, get_loss=self.get_loss)
            out_tr["tr"] = {}
            for mname in metrics.keys():
                total_loss = sum([out_tr[l_name][mname][0] for l_name in tr_names])
                total_inst = sum([out_tr[l_name][mname][1] for l_name in tr_names])
                out_tr["tr"][mname] = [total_loss, total_inst]
            if self.dataset_names is None:
                out_tr.pop("dummy")

            out.update(out_tr)

        if during_train:
            # use the stored values
            for tr_loader_key in self.current_train_loss.keys():
                try:
                    value = self.current_train_loss[tr_loader_key][0].cpu().item()
                except:
                    value = self.current_train_loss[tr_loader_key][0]
                num_inst = self.current_train_loss[tr_loader_key][1]
                out[tr_loader_key]={self.loss_name: [value, num_inst] }
        

        for l_name in out.keys():
            for m_name in out[l_name].keys():
                if average:
                    if out[l_name][m_name][1]>0.:
                        # perform the average
                        out[l_name][m_name] = out[l_name][m_name][0] / out[l_name][m_name][1]
                    else:
                        out[l_name][m_name] = float("nan")

        return out


    # returns a dictionary mapping dataloader type to dictionary that maps metric_names to metric values.
    def get_test_metrics(self, model, test_loaders=None, metrics=None, test_dataset_names=None):
        if test_loaders is None:
            test_loaders = self.test_loaders
            assert not self.test_loaders is None
        if not isinstance(test_loaders, list):
            test_loaders = [test_loaders]
        if test_dataset_names is None and len(test_loaders)>1:
            test_dataset_names = [str(i) + "te" for i in range(len(test_loaders))]
        elif test_dataset_names is None:
            test_dataset_names = ["te"]
        
        return self.get_metrics(model, metrics, additional_loaders=test_loaders, add_dataset_names=test_dataset_names, average=True)

    def write_params(self, path=None, nomore_eval=True):
        if path is None:
            path = self.version_path
        if nomore_eval:
            self.params["do_eval"] = 0
        if self.params != self.old_params:
            with open(os.path.join(path, "params.json"), "w") as f:
                json.dump(self.params, f, sort_keys=True, indent=2)
            self.old_params = self.params

    def read_params(self, path=None):
        if path is None:
            path = self.version_path
        with open(os.path.join(path, "params.json"), "r") as f:
            self.params = json.load(f)


    # append the logfile by the last entries of the metric_data dictionary
    def log(self, input=None):
        if input is None:
            logstr = self.get_log_string()
        else:
            logstr = input
        with open(os.path.join(self.version_path, "log.txt"), "a") as f:
            f.write("\n" + logstr)

    def get_memory_per_batch(self, model):
        infos = []
        optimizer = torch.optim.Adam(lr=0, params=model.parameters())
        for loader in self.train_loaders:
            batch = next(iter(loader))
            optimizer.zero_grad()
            y1, y2 = self.get_targets(model=model, batch=batch, device=self.device)
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            f = r-a
            mem_info = [r,a,f]
            infos.append(mem_info)
            del y1
            del y2
            del batch
        optimizer.zero_grad()
        return infos
        

    # read version name from files already present, depending on self.overwrite
    def set_version_counter(self):
        pass

    def get_log_string(self):
        logstr = "{:<4d}  ".format(self.epoch)
        logstr += self.get_metric_string() + "{:<.2E}".format(self.params["lr"])
        now = datetime.datetime.now()
        now.strftime("%H:%M:%S")
        logstr += "  " + str(now)
        if self.show_times:
            logstr += "  {:<.2E}".format(self.load_time)
            logstr += "  {:<.2E}".format(self.forward_time)
            logstr += "  {:<.2E}".format(self.backward_time)
        self.load_time = 0
        self.forward_time = 0 
        self.backward_time = 0
        
        return logstr

    def get_log_header(self):
        header = "epoch " + self.get_metric_header() + "lr        " + "time                        "
        if self.show_times:
            header += "load      forward   backward"
        return header

    def get_metric_string(self):
        metric_str = ""
        names = self.get_sorted_metric_data_names()
        for name in names:
            if name in self.do_not_show:
                continue
            try:
                # last entry of the respective list
                val = self.metric_data[name][-1]
                if np.isnan(val):
                    metric_str += "---nan---  "
                else:
                    metric_str += "{:<.2E}   ".format(val)
            except IndexError:
                # if the metric_data is still empty, do not throw an error
                metric_str += "--wait--   "
        return metric_str

    def get_metric_header(self):
        header = ""
        names = self.get_sorted_metric_data_names()
        for metric_name in names:
            # maximum extension is: "2.18E+00  "
            if len(metric_name) > 10:
                header += metric_name[:9]+" "
            else:
                header += metric_name + " "*(11-len(metric_name))
        return header

    def get_sorted_metric_data_names(self):
        # place train loss at the front:
        out = [self.loss_name+"_tr"]
        for name in self.metric_data.keys():
            if name in ["epoch", self.loss_name+"_tr"] or name in self.do_not_show:
                continue
            out.append(name)
        return out

    def get_print_string(self, batch_idx):
        metric_str = self.get_metric_string()
        # remove train loss part:
        metric_str = metric_str[10:]
        num_batches = sum([len(l) for l in self.train_loaders])
        curr_train_loss = self.current_train_loss["tr"][0] / self.current_train_loss["tr"][1]
        out_str = "{:<4d}   {:<4d}/{:<4d}  {:<.2E}  ".format(self.epoch, batch_idx, num_batches, curr_train_loss)
        out_str += metric_str
        return out_str

    def get_print_header(self):
        header = ""
        lossname = self.loss_name
        if len(lossname) > 7:
            lossname = lossname[:6]
        #tot leng "0      99  /99     2.18E+00   "
        header += "epoch  batch      "
        header += self.get_metric_header()
        return header

    def print_batch_update(self, batch_idx):
        out_str = self.get_print_string(batch_idx=batch_idx)
        print(out_str, end="\r")

    def clear(self):
        if os.path.exists(os.path.join(self.version_path, "log.txt")):
            open(os.path.join(self.version_path, "log.txt"), "w").close()
        if os.path.exists(os.path.join(self.version_path, "evaluation")):
            shutil.rmtree(os.path.join(self.version_path, "evaluation"))

    def _make_loss_figure(self, show=False, path=None):
        try:
            if path is None:
                path=os.path.join(self.version_path, "train_info")
            Train.make_loss_figure(y_names=[self.loss_name + "_tr", self.loss_name + "_vl"], metric_data=self.metric_data, folder_path=path, labels=["train", "val"], ylabel="Loss: "+self.loss_name, show=show)
            if not self.dataset_names is None:
                for ln in self.dataset_names:
                    Train.make_loss_figure(y_names=[self.loss_name + "_tr_" + ln, self.loss_name + "_vl_" + ln], metric_data=self.metric_data, folder_path=path, labels=["train", "val"], ylabel="Loss: "+self.loss_name, title="Learning Curve for dataset "+ln, name="training_" + ln, show=show)
        except:
            pass

    @staticmethod
    def make_loss_figure(y_names, folder_path="", metric_data=None, labels=None, ylabel=None, xlog=True, ylog=True, name="training", title="Learning Curve", show=False):
        if metric_data is None:
            with open(os.path.join(folder_path, "metric_data.json"), "r") as f:
                metric_data = json.load(f)
        if labels is None:
            labels = y_names
        if ylabel is None:
            ylabel = "Loss"

        fig, ax = plt.subplots()

        epochs = np.array(metric_data["epoch"], dtype=np.int32) + 1 #for log scaling
        for i, y_name in enumerate(y_names):
            ax.plot(epochs, np.array(metric_data[y_name]), label=labels[i])

        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.title(title)

        fig.savefig(os.path.join(folder_path, name+".png"))
        if show:
            plt.show()
        else:
            plt.close(fig)


    # returns a dict mapping loader names to a dict of 1d torch tensors containing targets
    @staticmethod
    def get_target_dict(model, loaders, get_targets, dataset_names=None, model_device="cuda", out_device="cpu", factor=1., target_name="y"):
        assert not model_device is None
        if not isinstance(loaders, list):
            loaders = [loaders]
        if dataset_names is None:
            if len(loaders)==1:
                dataset_names = [""]
            else:
                dataset_names = [str(i) for i in range(len(loaders))]
        assert len(loaders) == len(dataset_names)
        out = {}
        for i, loader in enumerate(loaders):
            out[dataset_names[i]] = {}
        model = model.to(model_device)
        for i, loader in enumerate(loaders):
            out[dataset_names[i]][target_name+"_pred"] = torch.zeros(0, device=out_device)
            out[dataset_names[i]][target_name+"_true"] = torch.zeros(0, device=out_device)
            with torch.no_grad():
                for batch in loader:
                    y_pred, y = get_targets(model=model, batch=batch, device=model_device)
                    out[dataset_names[i]][target_name+"_pred"] = torch.cat((out[dataset_names[i]][target_name+"_pred"], (y_pred.flatten()*factor).to(out_device)))
                    out[dataset_names[i]][target_name+"_true"] = torch.cat((out[dataset_names[i]][target_name+"_true"], (y.flatten()*factor).to(out_device)))
        return out

    @staticmethod
    def get_metrics_from_dict(target_dict, metrics, target_name="y", lname=None):
        out = {}
        for m_name in metrics.keys():
            if m_name == "loss": # loss is handled differently
                continue
            out[m_name] = {}
            if lname is None:
                lnames = list(target_dict.keys())
            else:
                lnames = [lname]

            for l_name in lnames:
                out[m_name][l_name] = metrics[m_name](target_dict[l_name][target_name+"_pred"], target_dict[l_name][target_name+"_true"]).item()
        return out


    # y_true is on x-axis, y_pred on y-axis
    @staticmethod
    def visualize_targets(y_true, y_pred, min_y=None, max_y=None, show=False, err_histogram=None, bins_err=50, bins_2d=200, name="y", ylabel=None, folder_path="", title_name=None, metric_dict=None, loader_name=None, log_scale_accuracy=True, percentile=100, errors=False, density=False, show_all=True, xlabel=None):

        if title_name is None:
            title_name = name
        if ylabel is None:
            ylabel = "y"
        os.makedirs(folder_path, exist_ok=True)
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        
        max_error_plotted = None
        y_mean = None
        if min_y is None or max_y is None:
            y_mean = np.mean(y_true)    
            max_plotted_a = np.percentile(np.abs(y_true-y_mean), percentile)
            max_plotted_b = np.percentile(np.abs(y_pred-y_mean), percentile)
            max_error_plotted = max(max_plotted_a, max_plotted_b)

        if min_y is None:
            min_y = y_mean - max_error_plotted*1.05
        if max_y is None:
            max_y = y_mean + max_error_plotted*1.05


        fig, ax = plt.subplots(figsize=(8,8))
        plt.title("Accuracy for " + title_name)

        if density:
            if log_scale_accuracy:
                h = ax.hist2d(y_true, y_pred, range=((min_y,max_y),(min_y,max_y)), bins=bins_2d, cmap="Blues", norm=matplotlib.colors.LogNorm())
            else:
                h = ax.hist2d(y_true, y_pred, range=((min_y,max_y),(min_y,max_y)), bins=bins_2d, cmap="Blues")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.1)
            # make a colorbar to count occurences
            try:
                fig.colorbar(h[3], cax=cax, orientation='vertical')
            except ValueError:
                raise RuntimeError("This error might be due to an empty diagram, try to cover a larger energy space. name="+name + ", y_pred.shape="+str(y_pred.shape))
        
        else:
            if len(y_true) > 1000:
                h = ax.scatter(y_true, y_pred, color = "blue", linewidths=0.1, marker=".", s=1, alpha=1)
            h = ax.scatter(y_true, y_pred, color = "blue", linewidths=0.1, marker=".")
            if not show_all:
                ax.set_xlim((min_y,max_y))
                ax.set_ylim((min_y,max_y))


        id = np.array([min_y, max_y])
        ax.plot(id,id, color="orange")
        ax.set_xlim(min_y, max_y)
        ax.set_ylim(min_y, max_y)


        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if not metric_dict is None:
            assert not loader_name is None
            l_names = list(metric_dict.keys())
            # metric_info = ""
            # for l_name in l_names:
            #     metric_info += l_name + ": {:.2E}".format(metric_dict[l_name][loader_key])+"\n"
            # metric_info = metric_info[:-1] # remove the last newline
            # plt.text(0,0,metric_info)
            # NOTE: one could also think about adding this as a label to some plot and printing a metric
            labels = []
            for l_name in l_names:
                labels.append(l_name + ": {:.2E}".format(metric_dict[l_name][loader_name]))
            labels.append("var: {:.2E}".format(y_true.std()**2))

            handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * len(labels)

            # create the legend, supressing the blank space of the empty line symbol and the
            # padding between symbol and label by setting handlelength and handletextpad
            ax.legend(handles, labels, loc='best', fontsize='small', 
                    fancybox=True, framealpha=0.7, 
                    handlelength=0, handletextpad=0)
        fig.savefig(os.path.join(folder_path, name+"_accuracy.png"))

        if show:
            plt.show()
        else:
            plt.close(fig)

        if errors:
            
            fig, ax = plt.subplots()
            plt.title("Errors for " + title_name)
            # do histogram:
            errors = y_pred-y_true
            err = err_histogram
            y_mean = np.mean(y_true)
            if err is None:
                if max_error_plotted is None:
                    err = np.percentile(np.abs(y_pred-y_mean), percentile)
                else:
                    err = max_error_plotted

            ax.hist(errors, bins=bins_err, range=(-err, err))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
            ax.set_xlabel(r"$\delta$"+ylabel)
            ax.set_ylabel("occurences")
            fig.savefig(os.path.join(folder_path, name+"_errors.png"))

            if show:
                plt.show()
            else:
                plt.close(fig)

    def make_loss_dict(self, model=None, loaders=None, dataset_names=None, device=None, target_factor=None, save_targets=True, path=None):
        if model is None:
            raise RuntimeError("model is None")

        if device is None:
            device = self.device
        if target_factor is None:
            target_factor = self.target_factor

        if path is None:
            path=self.version_path

        os.makedirs(path, exist_ok=True)
        # create loader and loader names list
        if loaders is None:
            assert len(self.train_loaders)==len(self.val_loaders)
            dataset_names = []
            loaders= []
            to_load = [self.train_loaders, self.val_loaders]
            
            if not self.test_loaders is None:
                to_load.append(self.test_loaders)
                assert len(self.train_loaders)==len(self.test_loaders)
            for j, mem in enumerate(to_load):
                for i, l in enumerate(mem):
                    loaders.append(l)
                    if self.dataset_names is None:
                        dataset_names.append(["train", "val", "test"][j])
                    else:
                        dataset_names.append(["tr", "vl", "te"][j] + "_" + self.dataset_names[i])

        if dataset_names is None:
            dataset_names = [str(i) for i in range(loaders)]

        target_dict = Train.get_target_dict(model, loaders=loaders, get_targets=self.get_targets, dataset_names=dataset_names, model_device=device, out_device="cpu", factor=target_factor)

        if save_targets:
            with open(os.path.join(self.version_path, "evaluation", "targets.pickle"), "bw") as f:
                pickle.dump(target_dict, f)
        loss_data = Train.get_metrics_from_dict(target_dict=target_dict, metrics=self.metrics)
        with open(os.path.join(path, "losses.json"), "w") as f:
            json.dump(loss_data, f)

        return target_dict, loss_data


    def evaluate_model(self, model=None, show=False, loaders=None, dataset_names=None, device=None, target_factor=None, save_targets=True, path=None, for_all=False, ylabel="y", min_y=None, max_y=None, err_histogram=None, bins_err=50, bins_2d=200, log_scale_accuracy=True, percentile=100, errors=False, density=False, show_all=True):

        if target_factor is None:
            target_factor = self.target_factor

        if loaders is None:
            assert len(self.train_loaders)==len(self.val_loaders)
            dataset_names = []
            loaders= []
            to_load = [self.train_loaders, self.val_loaders]
            
            if not self.test_loaders is None:
                to_load.append(self.test_loaders)
                assert len(self.train_loaders)==len(self.test_loaders)
            for j, mem in enumerate(to_load):
                for i, l in enumerate(mem):
                    loaders.append(l)
                    if self.dataset_names is None:
                        dataset_names.append(["train", "val", "test"][j])
                    else:
                        dataset_names.append(["tr", "vl", "te"][j] + "_" + self.dataset_names[i])

        if dataset_names is None:
            dataset_names = [str(i) for i in range(loaders)]

        figpath = os.path.join(path, "evaluation")

        target_dict, loss_data = self.make_loss_dict(model=model, loaders=loaders, dataset_names=dataset_names, device=device, target_factor=target_factor, save_targets=save_targets, path=figpath)
        
        # do the plots
        # first, for the whole dataset:
        assert for_all == False # not implemented yet
        if for_all and (not len(self.val_loaders) == 1):
            pass

        for l_name in dataset_names:
            Train.visualize_targets(y_true=target_dict[l_name]["y_true"], y_pred=target_dict[l_name]["y_pred"], min_y=min_y, max_y=max_y, show=show, err_histogram=err_histogram, bins_err=bins_err, bins_2d=bins_2d, name=l_name, ylabel=ylabel, folder_path=figpath, title_name=None, metric_dict=loss_data, loader_name=l_name, log_scale_accuracy=log_scale_accuracy, percentile=percentile, errors=errors, density=density, show_all=show_all)
        
        self._make_loss_figure(show=show, path=figpath)




    @staticmethod
    def get_loaders(datasets, split = [8,1,1], batch_sizes=[5], shuffle=True, collate_fn=None):
        if not isinstance(datasets, list):
            datasets = [datasets]
        if not isinstance(batch_sizes, list):
            batch_sizes = [batch_sizes]*len(datasets)
        if len(batch_sizes)==1 and len(datasets)!=1:
            batch_sizes = batch_sizes*len(datasets)

        def get_split(partition, ds):
            n_data = len(ds)
            partition = [int(n_data * x / sum(partition)) for x in partition]
            splitted = []
            idx = 0
            for p_size in partition:
                splitted.append(ds[idx : idx + p_size])
                idx += p_size
            return splitted

        loaders = [[] for _ in split]
        for i, dataset in enumerate(datasets):
            splitted = get_split(split, dataset)
            for j, ds in enumerate(splitted):
                loaders[j].append( torch.utils.data.DataLoader(ds, batch_size=batch_sizes[i], shuffle=shuffle, collate_fn=collate_fn) )
        return loaders

    def get_dataset_info(self, loader):
        out = {"total_instances":0}
        for batch in loader:
            num_inst += self.get_num_instances(batch)
            out["total_instances"] += float(num_inst)
        return out
    
    def memory_information(self, model):
        self.target_for_memory = True
        infos = self.get_memory_per_batch(model=model)
        self.target_for_memory = False
        device_info = "device: " + self.device + " --> memory:"
        for inf in infos:
            device_info += "\n" + "   reserved: {:<5d}MB,   alloc: {:<5d}MB,   free: {:<5d}MB".format(inf[0]//1028**2, inf[1]//1028**2, inf[2]//1028**2)
        print()
        print(device_info)
        self.log(device_info)

    def init_metric_data(self):
        self.metric_data = {"epoch":[]}
        for metric in list(self.metrics.keys()):
            name = metric + "_vl"
            self.metric_data[name] = []
            if not self.dataset_names is None:
                for l in self.dataset_names:
                    name = metric + "_vl_" + l
                    self.metric_data[name] = []
        self.metric_data[self.loss_name+"_tr"] = []
        if not self.dataset_names is None:
            for l in self.dataset_names:
                name = self.loss_name + "_tr_" + l
                self.metric_data[name] = []
