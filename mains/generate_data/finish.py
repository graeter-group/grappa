from pathlib import Path
import shutil
from single_points import calc_states
from validate_qm import validate_qm_data

def finish_run(folder, memory=28, num_threads=8):
    folder = Path(folder)
    # all subfolders of folder:
    subfolders = [f for f in folder.iterdir() if f.is_dir()]
    for subfolder in subfolders:
        # if no psi4_energies.npy in file, delete the directory:
        if not (subfolder/"psi4_energies.npy").exists():
            shutil.rmtree(subfolder)
        else:
            calc_states(subfolder, n_states=None, memory=memory, num_threads=num_threads)
        
    validate_qm_data(folder)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='The folder containing the PDB files.')
    parser.add_argument('--memory', '-m', type=int, help='The amount of memory to use.', default=32)
    parser.add_argument('--num_threads', '-t', type=int, help='The number of threads to use.', default=8)
    args = parser.parse_args()

    if args.memory < 0:
        args.memory = None
    if args.num_threads < 0:
        args.num_threads = None
    finish_run(folder=args.folder, memory=args.memory, num_threads=args.num_threads)