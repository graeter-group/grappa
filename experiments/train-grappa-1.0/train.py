if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="grappa-1.0", help="Project name for wandb.")
    parser.add_argument("-tb", "--train_batch", type=int, default=-1, help="Batch size for training.")
    parser.add_argument("-vb", "--val_batch", type=int, default=-1, help="Batch size for validation.")
    # parser.add_argument("--pretrain_path", type=str, default=None, help="Path to pretrained model.") #NOTE: include this arg
    parser.add_argument("-p", "--param_weight", type=float, default=None, help="Weight for the param loss of the datasets with classical parameters from amber99sbildn. Default is None.")

    args = parser.parse_args()


    from grappa.training.trainrun import do_trainrun
    import yaml
    from pathlib import Path
    import numpy as np

    # load the config:
    config_path = "grappa_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # set the splitpath:    
    config["data_config"]["splitpath"] = str(Path(__file__).parent.parent.parent/f"dataset_creation/get_espaloma_split/espaloma_split.json")

    # set the name:
    config["trainer_config"]["name"] = ""

    # set the batch sizes:
    if args.train_batch > 0:
        config["data_config"]["train_batch_size"] = args.train_batch
    if args.val_batch > 0:
        config["data_config"]["val_batch_size"] = args.val_batch

    if not args.param_weight is None:
        config["trainer_config"]["name"] += f"_p{int(np.log10(args.param_weight))}"
        config['lit_model_config']['param_weights_by_dataset'] = {ds: args.param_weight for ds in config['data_config']['datasets'] if 'amber99sbildn' in ds}


    # train:
    do_trainrun(config=config, project=args.project)