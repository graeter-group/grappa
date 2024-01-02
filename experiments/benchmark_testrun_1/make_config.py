import yaml
from pathlib import Path

CONFIG_PATH = "/hits/fast/mbm/seutelf/grappa/experiments/hyperparameter_optimization/wandb/run-20240101_154827-wl44wra8/files/grappa_config.yaml"

datasets = [
    "spice-dipeptide",
    "spice-des-monomers",
    "spice-pubchem",
    "gen2",
    "gen2-torsion",
    "pepconf-dlc",
    "protein-torsion",
    "rna-diverse",
]
pure_test_datasets = ["rna-trinucleotide"]
pure_train_datasets = ["rna-nucleoside"]

# Load the config
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# overwrite the data part of the config:
config["data_config"]["datasets"] = datasets
config["data_config"]["pure_test_datasets"] = pure_test_datasets
config["data_config"]["pure_train_datasets"] = pure_train_datasets


# save the config
with open(Path(__file__).parent / "grappa_config.yaml", "w") as f:
    yaml.dump(config, f)