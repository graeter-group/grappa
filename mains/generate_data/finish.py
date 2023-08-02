from pathlib import Path
import shutil
from single_points import calc_states
from validate_qm import validate_qm_data
import time

def finish_run(folder, memory=28, num_threads=8, delete=False):
    folder = Path(folder)
    # all subfolders and subfolders of subfolders that do not contain other folders:
    subfolders = [x for x in folder.glob("**/*") if x.is_dir() and not any([y.is_dir() for y in x.glob("*")])]
    removelist = []
    calclist = []
    for subfolder in subfolders:
        # if no psi4_energies.npy in file, delete the directory:
        if not (subfolder/"psi4_energies.npy").exists():
            removelist.append(subfolder)
        else:
            calclist.append(subfolder)

    # ask user if the folder should be deleted:
    if len(removelist) > 0:
        print("The following folders do not contain any QM data and will be deleted:")
        for i, folder in enumerate(removelist):
            print(f"{i}: {folder}")
        if not delete:
            print("Do you want to continue? (y/n)")
            answer = input()
        else:
            answer = "y"
        if answer == "y":
            for folder in removelist:
                shutil.rmtree(folder)
        else:
            print("Aborting.")
            return

    
    for subfolder in calclist:
        calc_states(subfolder, n_states=None, memory=memory, num_threads=num_threads)
        
    validate_qm_data(folder)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='The folder containing the PDB files.')
    parser.add_argument('--memory', '-m', type=int, help='The amount of memory to use.', default=32)
    parser.add_argument('--num_threads', '-t', type=int, help='The number of threads to use.', default=8)
    parser.add_argument('--delete', '-d', action='store_true', help='Delete folders without QM data unasked.', default=False)
    args = parser.parse_args()

    finish_run(folder=args.folder, memory=args.memory, num_threads=args.num_threads, delete=args.delete)