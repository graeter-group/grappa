from pathlib import Path
import os

this_dir = Path(__file__).parent

args = [
    ['no_gnn_attention'],
    ['no_gnn', 'with_hybridization'],
    ['no_self_interaction'],
    ['no_param_attention'],
    ['no_positional_encoding'],
    ['no_gated_torsion'],
    ['wrong_symmetry'],
    # ['no_scaling'],
    # ['exp_to_range'],
]




for i, arglist in enumerate(args):
    print(f'Job {i+1}')
    CMD_BASE = f"sbatch job.sh {this_dir}"
    CMD = f'python train.py --project ablation-grappa-1.0'
    for arg in arglist:
        CMD += f' --{arg}'
    print(f'Command: {CMD}')
    os.system(f'{CMD_BASE} {CMD}')