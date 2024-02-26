from grappa.training.resume_trainrun import resume_trainrun
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, help="Run id of the run to resume.", required=True)
parser.add_argument("--project", type=str, default="grappa-1.1", help="Project name for wandb.")

if __name__ == "__main__":

    args = parser.parse_args()

    def transform_config(config):
        config["lit_model_config"]["time_limit"] += 24
        return config

    resume_trainrun(run_id=args.run_id, project=args.project, new_wandb_run=False, transform_config=transform_config)