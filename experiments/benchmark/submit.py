from pathlib import Path
import os

this_dir = Path(__file__).parent

for i in range(5):
    print(f'Job {i+1}')
    CMD_BASE = f"sbatch job.sh {this_dir}"
    CMD = f'python train.py --project benchmark-grappa-1.0'
    if i==1:
        CMD += ' --with_hybridization'
    if i==2:
        CMD += ' -s 2 -o 2'
    if i==3:
        CMD += ' -s 4 -o 4'
    if i==4:
        CMD += ' -s 0.5 -o 0.5'
    print(f'Command: {CMD}')
    os.system(f'{CMD_BASE} {CMD}')