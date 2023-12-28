import time
from pathlib import Path
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--continue_finished', action='store_true', help='continue all finished runs with runtime > 10 hours')

args = parser.parse_args()


CMD = 'python resume.py' if not args.continue_finished else 'python resume.py --continue_finished'
N_JOBS = 5


# asser current path is the parent directory of this file
assert Path.cwd() == Path(__file__).parent

(Path.cwd()/'logs').mkdir(exist_ok=True)

for i in range(N_JOBS):
    print(f'Job {i+1}/{N_JOBS}')
    print(f'Command: {CMD}')
    os.system(f'sbatch job.sh "{CMD}"')