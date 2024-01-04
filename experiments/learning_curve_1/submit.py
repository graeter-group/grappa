from pathlib import Path
import os
import argparse

# assert current path is the parent directory of this file
assert Path.cwd() == Path(__file__).parent

(Path.cwd()/'logs').mkdir(exist_ok=True)

for k in [3]:#, 4, 6, 8]:

    for i in range(k):
        print(f'Job {k}-{i+1}')
        CMD = f'python train.py {i} {k} --project learning_curve_1'
        print(f'Command: {CMD}')
        os.system(f'sbatch job_lc.sh "{CMD}"')