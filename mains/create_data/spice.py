#%%

if __name__=="__main__":

    import argparse
    from grappa.constants import SPICEPATH
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--spicepath", type=str, default=SPICEPATH, help="Path to full spice hdf5 file.")
    parser.add_argument("--dipeppath", type=str, default=None, help="Path to dipeptide spice hdf5 file.")
    parser.add_argument("--smiles", action="store_true", help="Whether to include smiles in dataset.")
    parser.add_argument("--n_max", type=int, default=None, help="Maximum number of molecules to include in dataset.")
    parser.add_argument("--max_energy", type=float, default=None, help="Maximum energy difference between min and max energy state of a molecule to include in dataset, in grappa units")
    parser.add_argument("--max_force", type=float, default=None, help="Maximum force to include in dataset, in grappa units")
    parser.add_argument("--name", type=str, default=None, help="Name of dataset, default: spice if smiles=False, spice_openff if smiles=True")
    parser.add_argument("--not_skip_errs", action="store_false", dest="skip_errs", default=True, help="Whether to skip errors.")
    parser.add_argument("--skip_names", type=str, nargs="+", help="list of names of molecules to skip.")

    args = parser.parse_args()
    
    N_MAX = args.n_max
    SMILES = args.smiles
    max_energy = args.max_energy
    max_force = args.max_force
    name = args.name

    spicepath = args.spicepath

    dipeppath = str(Path(spicepath).parent/Path("dipeptides_spice.hdf5")) if args.dipeppath is None else args.dipeppath

    from grappa.PDBData.PDBDataset import PDBDataset
    from pathlib import Path
    import shutil
    import os
    from grappa.constants import SPICEPATH as spicepath


    if name is None:
        name = "spice" if not SMILES else "spice_openff"

    storepath = Path(spicepath).parent/Path(f"PDBDatasets/{name}/base")


    if os.path.exists(str(storepath)):
        shutil.rmtree(str(storepath))

    ds = PDBDataset.from_spice(dipeppath, info=True, n_max=N_MAX, with_smiles=SMILES, randomize=True, seed=0, skip_errs=args.skip_errs, skip_names=args.skip_names)

    # remove conformations with energy > 200 kcal/mol from min energy in ds[i]

    if max_energy is not None:
        ds.filter_confs(max_energy=max_energy, max_force=max_force, reference=False) # remove crashed conformations

    # delete folder is present:
    if os.path.exists(str(storepath)):
        shutil.rmtree(str(storepath))

    ds.save_npz(storepath, overwrite=True)

# %%
