from pathlib import Path
import os
import argparse

# assert current path is the parent directory of this file
assert Path.cwd() == Path(__file__).parent

this_dir = Path(__file__).parent

param_weights = [1e3, 1e1, 1e-1, 1e-3, 1e-5]

for i, param_weight in enumerate(param_weights):
    print(f'Job {i+1}')
    CMD_BASE = f"sbatch job.sh {this_dir}"
    CMD = f'python train.py --project grappa-1.0 -r -p {param_weight}'

    print(f'Command: {CMD}')
    os.system(f'{CMD_BASE} {CMD}')