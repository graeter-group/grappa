if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="grappa-1.0", help="Project name for wandb.")
    parser.add_argument("-r", "--radical", action="store_true", help="Use radical dataset.")
    parser.add_argument("-tb", "--train_batch", type=int, default=-1, help="Batch size for training.")
    parser.add_argument("-vb", "--val_batch", type=int, default=-1, help="Batch size for validation.")
    parser.add_argument("--with_hybridization", action="store_true", help="Use hybridization as input feature.")
    parser.add_argument("--rad-flag", action="store_true", help="Use the is_radical feature.")

    args = parser.parse_args()


    from grappa.training.trainrun import do_trainrun
    import yaml
    from pathlib import Path

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

    if args.with_hybridization:
        assert not args.radical, "Cannot use hybridization feature for radicals."
        config["model_config"]["in_feat_name"] += ["sp_hybridization"]
        config["trainer_config"]["name"] += "_hybrid"

    if args.radical:
        config["data_config"]["datasets"].append("dipeptide_rad")

    if args.rad_flag:
        if "is_radical" in config["model_config"]["in_feat_name"]:
            config["model_config"]["in_feat_name"].append("is_radical")
            config["trainer_config"]["name"] += "_rad_flag"

    # train:
    do_trainrun(config=config, project=args.project)