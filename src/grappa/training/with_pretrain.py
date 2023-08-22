#%%

# define some defaults
DEVICE = "cuda"
BCEWEIGHT = 100

import os
import shutil

import torch
from . import grappa_training, utilities
from ..run.eval_utils import evaluate
import json
import pandas as pd
pd.options.display.float_format = '{:.2E}'.format

import time
from pathlib import Path
from .grappa_training import RMSE

#%%

def train_with_pretrain(model, version_name, pretrain_name, tr_loader, vl_loader, lr_pre=1e-4, lr_conti=1e-4, energy_factor=0, force_factor=1, storage_path="versions", classification_epochs=-1, pretrain_epochs=10, epochs=500, patience=1e3, time_limit=2, device=DEVICE, bce_weight=BCEWEIGHT, pretrain_direct_epochs=100, direct_eval=False, param_statistics=None, param_factor=0.1, final_eval=True, reduce_factor=None, load_path=None, recover_optimizer=False, continue_path=None, use_warmup=False, weight_decay=0, scale_dict={}, l2_dict={}):
    """
    This function is neither written efficiently, nor well documented or tested. Only to be used for internal testing.
    load_path: path to the version directory of the model to continue from.
    """

    time_limit = int(time_limit*60*60) # convert hours to seconds

    model = model.to(device)

    if not reduce_factor is None:
        raise NotImplementedError("reduce_factor is deprecated.")

    optimizer = torch.optim.Adam(lr=lr_pre, params=model.parameters(), weight_decay=weight_decay)

    # only do load and passed load_pretrained is true
    load = not (load_path is None and continue_path is None)

    if (not load) and (pretrain_epochs > 0):
        trainer = grappa_training.TrainSequentialParams(energy_factor=0., force_factor=0., direct_epochs=pretrain_direct_epochs, train_loader=tr_loader, val_loader=vl_loader, print_interval=1, log_interval=1, figure_update_interval=None, batch_print_interval=25, evaluation_metrics={}, model_saving_interval=5, store_states=True,
        energies=["bond", "angle", "torsion", "improper", "bonded", "bonded_averaged", "ref", "reference_ff"],
        levels=["n2", "n3", "n4", "n4_improper"],
        clip_value=1e-1, average=True, reference_energy="u_ref", classification_epochs=classification_epochs, bce_weight=bce_weight, storage_path=os.path.join(storage_path,version_name), eval_forces=True, param_statistics=param_statistics, param_factor=param_factor, scale_dict=scale_dict, l2_dict=l2_dict, eval_train=False)

        #%%
        print("starting pretraining for version", pretrain_name, "\n")

        model = trainer.run(model=model, lr=lr_pre, epochs=pretrain_epochs, version_name=pretrain_name, device=device, log_note="", use_scheduler=False, saving=True,
        forced_optimizer=optimizer,
        final_eval=direct_eval,
        use_warmup=False,
        )
    
    if epochs > 0:

        # either continue from a previous run or from the pretrained model or continue or from scratch (if both are None)
        model_path = None
        assert (continue_path is None or load_path is None), "either continue_path or load_path has to be specified"
        if not continue_path is None:

            model_path = os.path.join(continue_path,"best_model.pt")
            if not os.path.exists(model_path):
                model_path = os.path.join(continue_path,"last_model.pt")

            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                print("continuing training of model from ", model_path, "\n")

        elif not load_path is None:

            model_path = os.path.join(load_path,"best_model.pt")
            if not os.path.exists(model_path):
                model_path = os.path.join(load_path,"last_model.pt")

            model.load_state_dict(torch.load(model_path, map_location=device))
            print("continuing training of model from ", model_path, "\n")
        else:
            if (not pretrain_name is None) and (pretrain_epochs > 0):
                model_path = os.path.join(storage_path,version_name,pretrain_name,"last_model.pt")

                model.load_state_dict(torch.load(model_path, map_location=device))
                print("loaded pretrained model from ", model_path, "\n")

        if not model_path is None:
            # do evaluation on the validation set:
            model = model.to(device)
            model.eval()
            print(f"\n\nevaluating model\n\t'{model_path}'\non tr and val set...\n")
            on_forces = False if force_factor == 0 else True
            eval_data = evaluate(loaders=[tr_loader, vl_loader], loader_names=["tr", "val"], model=model, device=device, on_forces=on_forces, rmse_plots=False)
            model.train()
    

            with open(os.path.join(storage_path,version_name,"pretrain_eval_data.json"), "w") as f:
                json.dump(eval_data, f, indent=4)
            print()
            print(str(pd.DataFrame(eval_data["eval_data"])))
            print()

        optimizer = torch.optim.Adam(lr=lr_conti, params=model.parameters(), weight_decay=weight_decay)

        if not model_path is None and recover_optimizer:
            optimizer.load_state_dict( torch.load( str(Path(model_path).parent/Path("best_opt.pt")), map_location=device ) )
            use_warmup=False

        # rename:
        if not continue_path is None:
            rename_files = [f for f in Path(continue_path).glob("*.[txt pt]*")]
            for file in rename_files:
                if len(file.name.split("-")) != 1:
                    k = int(file.name.split("-")[0])
                    new_name = str(k+1) + "-" + "".join(file.name.split("-")[1:])
                else:
                    new_name = "1-" + file.name
                shutil.copy(str(file), os.path.join(str(file.parent), new_name))

        class ScheduledTrainer(grappa_training.TrainSequentialParams):
            def get_scheduler(self, optimizer):
                # meaning of threshold: to make the epoch count as better than before it has to reduce the loss by this amount, i.e. large threshold leads to early reduction in lr
                # in this setting "better" means lowering the train loss by one percent
                # step is called every log-interval on the total train loss

                return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=1e-10, patience=patience, threshold=1e-3, threshold_mode='rel', cooldown=0, min_lr=1e-20)


            def training_epoch(self, model, do_all=False):
                if time.time() - self.start > time_limit:
                    self.log("\ntime limit reached, stopping training\n")
                    print("\ntime limit reached, stopping training\n")
                    self.more_training = False

                if self.params["lr"] < 1e-10 and self.epoch > 10:
                    self.log("\nminimum lr reached, stopping training\n")
                    print("\nminimum lr reached, stopping training\n")
                    self.more_training = False

                return super().training_epoch(model=model, do_all=do_all)


        direct_epochs = -1
        if direct_eval:
            direct_epochs = 0

        metrics = {}

        trainer = ScheduledTrainer(energy_factor=energy_factor, force_factor=force_factor, direct_epochs=direct_epochs, train_loader=tr_loader, val_loader=vl_loader, print_interval=1, log_interval=1, figure_update_interval=None, batch_print_interval=25, evaluation_metrics=metrics, model_saving_interval=5, store_states=True,
        energies=["bond", "angle", "torsion", "improper", "bonded", "bonded_averaged", "ref", "reference_ff"],
        levels=["n2", "n3", "n4", "n4_improper"],
        clip_value=1e-1, average=True, reference_energy="u_ref", classification_epochs=-1, bce_weight=0., storage_path=storage_path, eval_forces=True, param_statistics=param_statistics, param_factor=param_factor, scale_dict=scale_dict, l2_dict=l2_dict, eval_train=False)

        trainer.start = time.time()

        print("starting training from pretrained model\nto version", version_name,"\n")
        
        # torch.autograd.set_detect_anomaly(True)


        model = trainer.run(model=model, lr=lr_conti, epochs=epochs, version_name=version_name, device=device, log_note="", use_scheduler=True, saving=True,
        forced_optimizer=optimizer,
        final_eval=final_eval,
        early_stopping_criterion="f_mae_vl",
        use_warmup=use_warmup,
        )
