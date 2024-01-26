from pathlib import Path
import os
import argparse

# assert current path is the parent directory of this file
assert Path.cwd() == Path(__file__).parent

(Path.cwd()/'logs').mkdir(exist_ok=True)

# start three jobs: one for grappa without radicals, one for grappa with missing-hydogen-type radicals, and one for grappa with hybridization feature for comparison

for i in range(2):
    print(f'Job {i+1}')
    CMD = f'python train.py --project test_opt-scan'
    if i==1:
        CMD += ' --with_hybridization'
    print(f'Command: {CMD}')
    os.system(f'sbatch job.sh "{CMD}"')