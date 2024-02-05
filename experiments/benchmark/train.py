if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="benchmark-grappa-1.0", help="Project name for wandb.")
    parser.add_argument("-tb", "--train_batch", type=int, default=-1, help="Batch size for training.")
    parser.add_argument("-vb", "--val_batch", type=int, default=-1, help="Batch size for validation.")
    parser.add_argument("--with_hybridization", action="store_true", help="Use hybridization as input feature.")
    parser.add_argument("-o", "--opt-weight", type=float, default=1., help="Sampling factor for the opt-datasets (gen2/pepconf-dlc).")
    parser.add_argument("-s", "--scan-weight", type=float, default=1., help="Sampling factor for the torsion-scan-datasets (gen2-torsion/protein-torsion).")
    parser.add_argument("-g", "--gradient-weight", type=float, default=None, help="Weight for the gradient loss. Default: 0.3")
    parser.add_argument("-pd", "--gnn_dropout", type=float, default=None, help="Dropout rate for the model. Default: 0.1")

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
        config["model_config"]["in_feat_name"] += ["sp_hybridization"]
        config["trainer_config"]["name"] += "_hybrid"

    # set the sampling factors:
    config["data_config"]["weights"]["opt"] = args.opt_weight
    config["data_config"]["weights"]["scan"] = args.scan_weight

    if args.opt_weight != 1.:
        config["trainer_config"]["name"] += f"_opt{int(args.opt_weight)}"

    if args.scan_weight != 1.:
        config["trainer_config"]["name"] += f"_scan{int(args.scan_weight)}"

    if not args.gradient_weight is None:
        config["lit_model_config"]["gradient_weight"] = args.gradient_weight
        config["trainer_config"]["name"] += f"_g{args.gradient_weight}"

    if not args.gnn_dropout is None:
        config["model_config"]["gnn_dropout_attention"] = args.gnn_dropout
        config["trainer_config"]["name"] += f"_d{args.gnn_dropout}"
    

    # train:
    do_trainrun(config=config, project=args.project)