if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="esp_split_hybrid", help="Project name for wandb.")

    args = parser.parse_args()


    from grappa.training.trainrun import do_trainrun
    import yaml
    from pathlib import Path

    # load the config:
    with open("grappa_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # set the splitpath:    
    config["data_config"]["splitpath"] = str(Path(__file__).parent.parent.parent/f"dataset_creation/get_espaloma_split/espaloma_split.json")

    # set the name:
    config["trainer_config"]["name"] = f"espaloma_run"

    # train:
    do_trainrun(config=config, project=args.project)