from grappa.training.resume_trainrun import resume_trainrun
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run_id", type=str, help="Run id of the run to resume.")
parser.add_argument("--project", type=str, default="grappa-1.1", help="Project name for wandb.")

args = parser.parse_args()

def transform_config(config):
    # remove all entries that include 'tripetide' in the dataset name:
    config['lit_model_config']['param_weights_by_dataset'] = {k:v for k,v in config['lit_model_config']['param_weights_by_dataset'].items() if 'tripeptide' not in k}
    return config

resume_trainrun(run_id=args.run_id, project=args.project, new_wandb_run=False, transform_config=transform_config)