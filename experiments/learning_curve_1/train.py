if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fold", type=int, help="Fold to train on. Runs 0-9.")
    parser.add_argument("k", type=int, help="Number of folds.")
    parser.add_argument("--project", type=str, default="learning_curve_1", help="Project name for wandb.")

    args = parser.parse_args()

    assert args.fold < args.k

    from grappa.training.trainrun import safe_trainrun
    import yaml
    from pathlib import Path

    # load the config:
    with open("grappa_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # set the splitpath:    
    config["data_config"]["splitpath"] = str(Path(__file__).parent/f"splits_{args.k}/split_{args.fold}.json")

    # set the name:
    config["trainer_config"]["name"] = f"k_{args.k}_fold_{args.fold}"

    # train:
    safe_trainrun(config=config, project=args.project)