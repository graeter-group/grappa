#%%
if __name__ == "__main__":
    from grappa.PDBData.PDBDataset import PDBDataset
    from openmm.app import ForceField

    import argparse
    from pathlib import Path
    from grappa.constants import DEFAULTBASEPATH


    parser = argparse.ArgumentParser(description="Parametrize a dataset with a charge model and force field. Will create a .bin file at ds_base/ds_name.parent / chargemodel_forcefield_filtered_dgl.bin")

    parser.add_argument("--ds_name", type=str, default="spice/charge_default_ff_amber99sbildn",  help="Name of the parametrized dataset to be loaded. Will load from the directory base_path/ds_name. Default: spice/charge_default_ff_amber99sbildn")

    parser.add_argument("--ds_base", type=str, default=str(Path(DEFAULTBASEPATH)), help=f"Base path to the dataset, default: {DEFAULTBASEPATH}")

    parser.add_argument("--overwrite", "-o", action="store_true", help="Overwrite existing files")

    parser.add_argument("--max_energy", "-e", type=float, default=62.5, help="Maximum energy difference between min and max energy state of a molecule to include in dataset, in grappa units. For the filtered dataset")

    parser.add_argument("--max_force", "-f", type=float, default=500, help="Maximum force to include in dataset, in grappa units. For the filtered dataset")

    args = parser.parse_args()

    loadpath = str(Path(args.ds_base)/Path(args.ds_name))

    storepath = loadpath + "_filtered"


    print("starting...")
    print("will load from ", loadpath)

    print("will store to ", storepath)

    print("will filter by energy < ", args.max_energy)
    print("will filter by force < ", args.max_force)

    ds = PDBDataset.load_npz(loadpath, n_max=None)

    # remove conformations with energy > x kcal/mol from min energy in ds[i]
    ds.filter_confs(max_energy=args.max_energy, max_force=args.max_force, reference=False)

    ds.save_npz(storepath, overwrite=args.overwrite)
    ds.save_dgl(storepath+"_dgl.bin", overwrite=args.overwrite)
        