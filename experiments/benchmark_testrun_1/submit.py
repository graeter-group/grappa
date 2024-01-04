from pathlib import Path
import os
import argparse

# assert current path is the parent directory of this file
assert Path.cwd() == Path(__file__).parent

(Path.cwd()/'logs').mkdir(exist_ok=True)

for i in range(10):
    print(f'Job {i+1}')
    CMD = f'python train.py {i} --project benchmark_testrun_1'
    print(f'Command: {CMD}')
    os.system(f'sbatch job_benchmark.sh "{CMD}"')