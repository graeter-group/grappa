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

    # with open('/hits/fast/mbm/seutelf/grappa/experiments/laplacian/wandb/run-20240125_021847-wm0njfrx/files/grappa_config.yaml', 'r') as f:
    #     config = yaml.safe_load(f)

        config['lit_model_config']['start_qm_epochs'] = 0

    # set the splitpath:    
    config["data_config"]["splitpath"] = str(Path(__file__).parent.parent.parent/f"dataset_creation/get_espaloma_split/espaloma_split.json")

    # set the name:
    config["trainer_config"]["name"] = f""

    continue_path = '/hits/fast/mbm/seutelf/grappa/experiments/laplacian/wandb/run-20240125_021847-wm0njfrx/files/checkpoints/last.ckpt'

    # train:
    do_trainrun(config=config, project=args.project, pretrain_path=continue_path)