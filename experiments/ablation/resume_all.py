#%%
from pathlib import Path
import os
import wandb

this_dir = Path(__file__).parent

PROJECT = 'ablation-grappa-1.0'

EXCEPT = []

# get all run ids:
api = wandb.Api()
runs = api.runs(PROJECT)
run_ids = [run.id for run in runs if run.state != 'running' and run.id not in EXCEPT]
#%%
for run_id in run_ids:
    print(f'Job {run_id}')
    CMD_BASE = f"sbatch job.sh {this_dir}"
    CMD = f'python resume.py {run_id} --project {PROJECT}'
    print(f'Command: {CMD}')
    os.system(f'{CMD_BASE} {CMD}')