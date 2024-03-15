from pathlib import Path
import os

this_dir = Path(__file__).parent

subsampling_factors = [0.75, 0.5, 0.25, 0.1, 0.05, 0.01]

for i, weight in enumerate(subsampling_factors):
    print(f'Job {i+1}')
    CMD_BASE = f"sbatch job.sh {this_dir}"
    CMD = f'python train.py --project benchmark-grappa-1.1-lc'
    CMD += f' --shrink_train {weight}'
    print(f'Command: {CMD}')
    os.system(f'{CMD_BASE} {CMD}')