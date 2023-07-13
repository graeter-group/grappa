# run to generate parametrized spice data

from grappa.PDBData.PDBDataset import PDBDataset
from pathlib import Path
from openmm.app import ForceField
from grappa.ff_utils.charge_models.charge_models import model_from_dict, randomize_model
import os
import warnings
warnings.filterwarnings("ignore") # for esp_charge

from grappa.constants import DEFAULTBASEPATH

FILTER_REF = True # whether to apply the filtering to the reference energies or the qm energies

def make_ds(get_charges, storepath, dspath, overwrite=False, allow_radicals=False, n_max=None, collagen=False, force_field="amber99sbildn.xml"):

    print("starting...")
    print("will write to ", storepath)

    ds = PDBDataset()

    ds = PDBDataset.load_npz(dspath, n_max=n_max)

    ff = ForceField(force_field)

    if not collagen:
        ds.remove_names(patterns=["HYP", "DOP"], upper=True)

    ds.parametrize(forcefield=ff, get_charges=get_charges, allow_radicals=allow_radicals, collagen=collagen)

    # filter out conformations that are way out of equilibrium:
    ds.filter_confs(max_energy=200, max_force=500, reference=FILTER_REF)

    
    ds.save_npz(storepath, overwrite=overwrite)
    ds.save_dgl(str(storepath)+"_dgl.bin", overwrite=overwrite, forcefield=ff, collagen=collagen, allow_radicals=allow_radicals)
    #%%

    ds.parametrize(forcefield=ff, get_charges=get_charges, allow_radicals=allow_radicals, collagen=collagen)
    
    # remove conformations with energy > 60 kcal/mol from min energy in ds[i]
    ds.filter_confs(max_energy=60, max_force=200, reference=FILTER_REF)

    # unfortunately, have to parametrize again to get the right shapes
    ds.save_npz(str(storepath)+"_60", overwrite=overwrite)
    ds.save_dgl(str(storepath)+"_60_dgl.bin", overwrite=overwrite, forcefield=ff, collagen=collagen, allow_radicals=allow_radicals)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="parametrize a dataset with a charge model and force field. will create a .bin file at ds_base/ds_name.parent / chargemodel_forcefield(_60)_dgl.bin")

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
                if args.collagen:
                    storepath += "_col"
                if not noise_level is None:
                    storepath += f"_{noise_level}"

                storepath += f"_ff_{args.force_field[:-4]}"

                if tag is not None:
                    get_charges = model_from_dict(tag=tag)
                else:
                    get_charges = None

                if not noise_level is None:
                    get_charges = randomize_model(model_from_dict(tag=tag), noise_level=noise_level)
                    print()
                    print(f"noise level: {noise_level}")

                make_ds(get_charges=get_charges, storepath=storepath, dspath=dspath, overwrite=args.overwrite, allow_radicals=args.allow_radicals, n_max=args.n_max, collagen=args.collagen, force_field=args.force_field)