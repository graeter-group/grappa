from pathlib import Path
import os

from train import parser

args = parser.parse_args()

# assert current path is the parent directory of this file
assert Path.cwd() == Path(__file__).parent

this_dir = Path(__file__).parent

CMD_BASE = f"sbatch job.sh {this_dir}"
CMD = f'python train.py'

for k, v in args.__dict__.items():
    if v is not None:
        if v == True:
            CMD += f' --{k}'
        elif v == False:
            pass
        else:
            CMD += f' --{k} {v}'

print(f'Command: {CMD}')
os.system(f'{CMD_BASE} {CMD}')