from pathlib import Path
import os
import argparse

# assert current path is the parent directory of this file
assert Path.cwd() == Path(__file__).parent

this_dir = Path(__file__).parent

for i in range(2):
    print(f'Job {i+1}')
    CMD_BASE = f"sbatch job.sh {this_dir}"
    CMD = f'python train.py --project grappa-1.1 -r'
    if i==1:
        CMD += ' --rad-flag'
    if i==2:
        CMD += ' --AA_bondbreak_rad'
    print(f'Command: {CMD}')
    os.system(f'{CMD_BASE} {CMD}')