#%%
from pathlib import Path
import os
import wandb

this_dir = Path(__file__).parent

PROJECT = 'grappa-1.0'

EXCEPT = ['b44wfpbw']

# get all run ids:
api = wandb.Api()
runs = api.runs(PROJECT)
run_ids = [run.id for run in runs]

#%%
for run_id in run_ids:
    if run_id in EXCEPT:
        continue
    print(f'Job {run_id}')
    CMD_BASE = f"sbatch job.sh {this_dir}"
    CMD = f'python resume.py {run_id} --project {PROJECT}'
    print(f'Command: {CMD}')
    os.system(f'{CMD_BASE} {CMD}')