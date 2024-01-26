if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="amber99_charges", help="Project name for wandb.")

    args = parser.parse_args()


    from grappa.training.trainrun import do_trainrun
    import yaml
    from pathlib import Path

    # load the config:
    with open("grappa_config_both.yaml", "r") as f:
        config = yaml.safe_load(f)

    # set the splitpath:    
    config["data_config"]["splitpath"] = str(Path(__file__).parent.parent.parent/f"dataset_creation/get_espaloma_split/espaloma_split.json")

    # set the name:
    config["trainer_config"]["name"] = f"both"

    # train:
    pretrain_path = Path(__file__).parent/"wandb/run-20240123_172812-wpv9wndk/files/checkpoints/best-model.ckpt"
    do_trainrun(config=config, project=args.project, pretrain_path=pretrain_path)