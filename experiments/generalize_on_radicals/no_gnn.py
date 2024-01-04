if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fold", type=int, help="Fold to train on. Runs 0-9.")
    parser.add_argument("--project", type=str, default="no_gnn", help="Project name for wandb.")

    args = parser.parse_args()


    from grappa.training.trainrun import safe_trainrun
    import yaml
    from pathlib import Path

    # load the config:
    with open("grappa_config_no_gnn.yaml", "r") as f:
        config = yaml.safe_load(f)

    # set the splitpath:    
    config["data_config"]["splitpath"] = str(Path(__file__).parent.parent/f"benchmark_testrun_1/splits/split_{args.fold}.json")

    # set the name:
    config["trainer_config"]["name"] = f"fold_{args.fold}"

    # train:
    safe_trainrun(config=config, project=args.project)