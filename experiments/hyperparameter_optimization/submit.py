import time
from pathlib import Path
import os

SWEEP_ID = 'el38p94o'

AGENT_CMD = f'wandb agent leif-seute/hpo_grappa/{SWEEP_ID}'
N_JOBS = 20

SLEEP_SECONDS = 0.01
# submit jobs using bash job.sh AGENT_CMD and wait a short time in between to avoid same-time communication with the wandb server

# asser current path is the parent directory of this file
assert Path.cwd() == Path(__file__).parent

(Path.cwd()/'logs').mkdir(exist_ok=True)

for i in range(N_JOBS):
    print(f'Job {i+1}/{N_JOBS}')
    print(f'Command: {AGENT_CMD}')
    os.system(f'sbatch job.sh "{AGENT_CMD}"')
    time.sleep(SLEEP_SECONDS)