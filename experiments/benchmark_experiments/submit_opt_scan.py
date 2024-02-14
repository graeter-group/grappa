from pathlib import Path
import os

this_dir = Path(__file__).parent

w = [10, 5, 3, 1, 0]

for i, weight in enumerate(w):
    print(f'Job {i+1}')
    CMD_BASE = f"sbatch job.sh {this_dir}"
    CMD = f'python train.py --project benchmark-grappa-1.0-exp'
    CMD += f' --opt-weight {weight} --scan-weight {weight}'
    print(f'Command: {CMD}')
    os.system(f'{CMD_BASE} {CMD}')