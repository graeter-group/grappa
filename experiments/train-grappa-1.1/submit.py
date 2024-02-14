from pathlib import Path
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--param_weight", type=float, default=None, help="Weight for the param loss of the datasets with classical parameters from amber99sbildn. Default is None.")
parser.add_argument("--bondbreak_radicals", '-b', action='store_true', default=False, help="Whether to include bond breaking radicals in the training set. Default is False.")

args = parser.parse_args()

# assert current path is the parent directory of this file
assert Path.cwd() == Path(__file__).parent

this_dir = Path(__file__).parent

CMD_BASE = f"sbatch job.sh {this_dir}"
CMD = f'python train.py --project grappa-1.1'

if not args.param_weight is None:
    CMD += f' --param_weight {args.param_weight}'
if args.bondbreak_radicals:
    CMD += ' --bondbreak_radicals'

print(f'Command: {CMD}')
os.system(f'{CMD_BASE} {CMD}')