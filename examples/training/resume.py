"""
Training a model on the spice-dipeptides dataset.
"""

from grappa.training.resume_trainrun import resume_trainrun
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run_id", type=str, help="Run id of the run to resume.")

args = parser.parse_args()

resume_trainrun(run_id=args.run_id, project="grappa_example", new_wandb_run=False)