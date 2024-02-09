from pathlib import Path
import os

this_dir = Path(__file__).parent

for i in range(3):
    print(f'Job {i+1}')
    CMD_BASE = f"sbatch job.sh {this_dir}"
    CMD = f'python train.py --project benchmark-grappa-1.0'
    print(f'Command: {CMD}')
    os.system(f'{CMD_BASE} {CMD}')