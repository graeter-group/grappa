# run to generate parametrized spice data


from pathlib import Path

import os


from grappa.ff_utils.charge_models.charge_models import model_from_dict, randomize_model
from grappa.constants import DEFAULTBASEPATH
from grappa.ff_utils.classical_ff.collagen_utility import get_mod_amber99sbildn

FILTER_REF = True # whether to apply the filtering to the reference energies or the qm energies

def make_ds(get_charges, storepath, dspath, overwrite=False, allow_radicals=False, n_max=None, collagen=False, force_field="amber99sbildn.xml", max_energy=None, max_force=None, openff_energies=False, name=None, recover=False, logpath=None):

    from grappa.PDBData.PDBDataset import PDBDataset
    from openmm.app import ForceField


    print("starting...")
    print("will write to ", storepath)

    ds = PDBDataset.load_npz(dspath, n_max=n_max)

    if not openff_energies:
        ff = ForceField(force_field)
    else:
        ff = force_field

    if not collagen:
        ds.remove_names(patterns=["HYP", "DOP"], upper=True)

    if collagen:
        ff = get_mod_amber99sbildn()

    # ds.parametrize(forcefield=ff, get_charges=get_charges, allow_radicals=allow_radicals, collagen=collagen, skip_errs=True, backup_path=storepath, recover=False)
    ds.parametrize(forcefield=ff, get_charges=get_charges, allow_radicals=allow_radicals, collagen=False, skip_errs=True, backup_path=storepath, recover=recover, logpath=logpath)

    # filter out conformations that are way out of equilibrium:
    ds.filter_confs(max_energy=200, max_force=500, reference=FILTER_REF)

    ds.save_npz(storepath, overwrite=overwrite)
    # ds.save_dgl(str(storepath)+"_dgl.bin", overwrite=overwrite)
    #%%

    
    # remove conformations with energy > x kcal/mol from min energy in ds[i]

    ds.filter_confs(max_energy=max_energy, max_force=max_force, reference=FILTER_REF)

    ds.save_npz(str(storepath)+"_filtered", overwrite=overwrite)


    # create evaluation plots:
    if not name is None:
        this_file = Path(__file__).parent
        plotdir = this_file/name
        ds.evaluate(plotpath=plotdir, suffix="_total_ref")





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="parametrize a dataset with a charge model and force field. will create a .bin file at ds_base/ds_name.parent / chargemodel_forcefield_filtered_dgl.bin")

    parser.add_argument("--ds_name", type=str, default=["spice/base"], nargs='+', help="name of the dataset to be loaded. will load from the directory base_path/ds_name default: spice/base")

    parser.add_argument("--ds_base", type=str, default=str(Path(DEFAULTBASEPATH)), help=f"base path to the dataset, default: {DEFAULTBASEPATH}")

    parser.add_argument("--charge", "-c", type=str, dest="tag", default=[None], nargs="+", help="tag of the charge model to use, see model_from_dict in charge_models.py.\npossible tags: ['bmk', 'avg', 'heavy, 'amber99sbildn'] . If None, call it default and use the charges of the force field.")

    parser.add_argument("--overwrite", "-o", action="store_true", default=False, help="overwrite existing .npz dataset, default:False")

    parser.add_argument("--noise_level", type=float, default=[None], nargs="+", help="noise level to add to the charges, default:None")

    parser.add_argument("--storage_dir", "-s", type=str, default=None, help="directory path relative to ds_base. in this folder, the parametrizes datasets are stored, named by the tag and noise_level. if None, this is the parent of ds_name. default: None")

    parser.add_argument("--allow_radicals", "-r", action="store_true", default=False)

    parser.add_argument("--n_max", type=int, default=None, help="maximum number of conformations to load from the dataset, default: None")

    parser.add_argument("--collagen", "-col", action="store_true", default=False, help="use collagen the forcefield instead of amber99sbildn. Will add '_col' to the name. Also remove any molecules with DOP or HYP in the name. default: False")

    parser.add_argument("--force_field", "-ff", default="amber99sbildn.xml", help="force field file (xml) to use for parametrization, default: amber99sbildn.xml.")

    parser.add_argument("--openff_energies", "-off", default=False, help="If openff force fields are used (such as gaff-2.11)", action="store_true")

    parser.add_argument("--max_energy", type=float, default=None, help="Maximum energy difference between min and max energy state of a molecule to include in dataset, in grappa units. For the filtered dataset")

    parser.add_argument("--max_force", type=float, default=None, help="Maximum force to include in dataset, in grappa units. For the filtered dataset")

    parser.add_argument("--recover", "-rec", action="store_true", default=False, help="recover from a previous run. will load the dataset from the storepath and continue parametrization. default: False")

    args = parser.parse_args()
    for ds_name in args.ds_name:
        for tag in args.tag:
            print()
            if tag is None:
                print("no additional charge model")
            else:
                print(f"charge tag: {tag}")

            for noise_level in args.noise_level:
                dspath = Path(args.ds_base)/Path(ds_name)
                storebase = str(dspath.parent) if args.storage_dir is None else str(Path(args.ds_base)/Path(args.storage_dir))
                
                if not tag is None:
                    tag_ = tag
                else:
                    tag_ = "default"

                storepath = os.path.join(str(storebase),"charge_"+tag_) # tagged with charge model

                logpath = Path(storepath).parent/"log.txt"
                logpath = str(logpath)

                if args.collagen:
                    storepath += "_col"
                if not noise_level is None:
                    storepath += f"_{noise_level}"

                if args.force_field[-4:] == ".xml":
                    storepath += f"_ff_{args.force_field[:-4]}"
                else:
                    storepath += f"_ff_{args.force_field}"

                if tag is not None:
                    get_charges = model_from_dict(tag=tag)
                else:
                    get_charges = None

                if not noise_level is None:
                    get_charges = randomize_model(model_from_dict(tag=tag), noise_level=noise_level)
                    print()
                    print(f"noise level: {noise_level}")

                storepath = storepath.replace(" ", "_")
                storepath = storepath.replace(".", "_")

                print(f"will write to {storepath}")
                print(f"starting the parametrization...")
                
                with open(logpath, "a") as f:
                    f.write(f"will write to {storepath}\n")
                    f.write(f"starting the parametrization...\n")


                make_ds(get_charges=get_charges, storepath=storepath, dspath=dspath, overwrite=args.overwrite, allow_radicals=args.allow_radicals, n_max=args.n_max, collagen=args.collagen, force_field=args.force_field, openff_energies=args.openff_energies, max_energy=args.max_energy, max_force=args.max_force, name=ds_name.split("/")[0], recover=args.recover, logpath=logpath)



# example usage for openff force fields: python make_graphs.py -off --ds_name spice_openff/base -ff gaff-2.11