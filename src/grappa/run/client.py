
import argparse
from pathlib import Path
from grappa.run.run import run_from_config
from grappa.constants import DEFAULTBASEPATH

def run_client():


    parser = argparse.ArgumentParser(
        epilog="If not using the ds_path argument, your dataset must be stored as dgl graphs in files named '{ds_base}/{ds_tag}_dgl.bin'.",
        description="Run a model from a config file. If no config file is specified, a default config file is created.")

    parser.add_argument('--run_config_path', type=str, default=None, help="path to a config file, used for default arguments. if not specified or no config file is found, creates a default config file. (default: None)")
    parser.add_argument('--model_config_path', type=str, default=None, help="path to a config file, used for default arguments. if not specified or no config file is found, creates a default config file. (default: None)")
    parser.add_argument('-p', '--ds_path', type=str, nargs='+', default=None, help="path to dataset. setting this overwrites the effect of ds_tag. (default: None)")
    parser.add_argument('--ds_base', type=str, default=DEFAULTBASEPATH, help=f"if tags are used, this is the base path to the datasets (default: {DEFAULTBASEPATH})")
    parser.add_argument('-t', '--ds_tag', type=str, nargs='+', default=[], help=" (dataset must be stored as dgl graphs in files named '{ds_base}/{ds_tag}_dgl.bin'. default: [])")
    parser.add_argument('--force_factor','-f', type=float, default=None, help=" (default: 1.)")
    parser.add_argument('--energy_factor','-e', type=float, default=None, help=" (default: 1.)")
    parser.add_argument('--recover_optimizer', action='store_true', default=False, help=" (if true, instead of a warm restart, we use the optimizer from the version specified by --load_path. default: False)")
    parser.add_argument('--warmup', action='store_true', default=False, help=" (if true, instead of a warm restart, we use the optimizer from the version specified by --load_path. default: False)")
    parser.add_argument('--continue_path', type=str, default=None, help="the version path of the version where we wish to continue training. the old logfile and models are stored, the rest is overwritten. (default: None)") 
    parser.add_argument('--param_weight', type=float, default=None, help=" (default: 0)")
    parser.add_argument('-d', '--description', type=str, nargs='+', default=[""], help='does nothing. only written in the config file for keeping track of several options that could be specified by hand like different datasets. (default: [""])')
    parser.add_argument('--load_path', type=str, default=None, help="path to the version-directory of the model to continue from. this overwrites the pretraining options, i.e. if both are specified, there is no pretraining. either absolute or relative to the path from where the command is being run. (default: None)")
    parser.add_argument('-c', '--confs', type=int, default=None, help=" (default: None)")
    parser.add_argument('-m', '--mols', type=int, default=None, help=" (default: None)")
    parser.add_argument('--storage_path', type=str, default=None, help="The path in which the version-directories are stored. (default: 'versions')")
    parser.add_argument('--name', type=str, default='', help="Optional: The name of the version directory (up to indices). (default: '')")
    parser.add_argument('--n', type=int, default=None, help="how many times to run (default: None)") # NOTE: this might be unnecesary now that we have several seeds
    parser.add_argument('--test', action='store_true', default=False, help="Reducing dataset size to 50 graphs for testing functionality. (default: False)'")
    parser.add_argument('--seed','-s', type=int, nargs='+', default=None, help="random seed for the split of the dataset into train, val, test (default: 0)")
    parser.add_argument('--pretrain_steps', type=float, default=None, help="approximate number of gradient updates for pretraining (default: 500)")
    parser.add_argument('--train_steps', type=float, default=None, help="approximate max number of gradient updates for training (default: 100000.0)")
    parser.add_argument('--patience', type=float, default=None, help="ratio of maximum total steps and patience of the learning rate scheduler, i.e. if (default: 0.0001)")
    parser.add_argument('--plots', action='store_true', default=False, help="create plots during and at the end of training. might take time and memory (default: False)")
    parser.add_argument('--ref_ff', type=str, default="amber99sbildn", help="suffix of the reference parameters for pretraining and plotting (default: 'amber99sbildn')")
    parser.add_argument('--lr', type=float, default=None, help="the learning rate (does not apply to pretraining on parameters) (default: '1e-6')")
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--ds_short', default=[], type=str, nargs="+", help="codes for a collections of datasets that are added to the ds_paths. available options: \n'eric_nat': with amber charges, energies filtered at 60kcal/mol, \n'eric_rad': with amber charges (heavy), energies filtered at 60kcal/mol, \n'spice': with energies filtered at 60kcal/mol and filtered for standard amino acids, \n'eric' both of the above (default: [])")
    parser.add_argument('--collagen', default=False, action="store_true", help="Whether or not to use the collagen forcefield and dataset, i.e. including hyp and dop. (default: False)")

    parser.add_argument('--n_conv', type=int, default=None, help=" (default: 3)")
    parser.add_argument('--n_att', type=int, default=None, help=" (default: 3)")
    parser.add_argument('--n_heads', type=int, default=None, help="Number of attention heads in the graph attention model. (default: 6)")
    parser.add_argument('--width', type=int, default=None, help=" (default: 512)")
    parser.add_argument('--rep_feats', type=int, default=None, help=" (default: 512)")
    parser.add_argument('--in_feat_name', type=str, nargs='+', default=None, help='which input features the model should use. (default: ["atomic_number", "residue", "in_ring", "formal_charge", "is_radical"])')
    parser.add_argument('--old_model', '-o', default=False, action="store_true", help="Whether or not to use the old model architecture (default: False)")

    args = parser.parse_args()

    for ds_short in args.ds_short:
        suffix = "_60"
        suffix_col = ""
        if args.collagen:
            suffix_col = "_col"
        if ds_short == "eric_nat":
            args.ds_tag += [f'AA_scan_nat/amber99sbildn{suffix_col}_amber99sbildn{suffix}', f'AA_opt_nat/amber99sbildn{suffix_col}']
        if ds_short == "eric_rad":
            args.ds_tag += [f'AA_scan_rad/heavy{suffix_col}_amber99sbildn{suffix}', f'AA_opt_rad/heavy{suffix_col}_amber99sbildn{suffix}']
        if ds_short == "spice":
            args.ds_tag += [f'spice/charge_default_ff_amber99sbildn{suffix}']
        if ds_short == "eric":
            args.ds_tag += [f'AA_scan_nat/amber99sbildn{suffix_col}_amber99sbildn{suffix}', f'AA_opt_nat/amber99sbildn{suffix_col}_amber99sbildn{suffix}', f'AA_scan_rad/heavy{suffix_col}_amber99sbildn{suffix}', f'AA_opt_rad/heavy{suffix_col}_amber99sbildn{suffix}']


    tags = None
    if args.ds_path is None:
        if args.ds_tag is None and args.run_config_path is None:
            raise ValueError("either ds_path or ds_tag or a config path must be specified")
    if len(args.ds_tag)>0:
        if args.ds_path is None:
            args.ds_path = []
        args.ds_path += [str(Path(f"{args.ds_base}/{tag}_dgl.bin")) for tag in args.ds_tag]

    if args.ds_tag == []:
        args.ds_tag = None
    tags = args.ds_tag

    if type(args.ds_path) == str:
        args.ds_path = [args.ds_path]

    seeds = args.seed
    if seeds is None:
        seeds = [0]

    run_config_path = args.run_config_path
    model_config_path = args.model_config_path

    assert args.continue_path is None or run_config_path is None, "cannot specify both continue_path and run_config_path"

    if not args.continue_path is None:
        run_config_path = f"{args.continue_path}/config.yaml"

    if not args.continue_path is None:
        model_config_path = f"{args.continue_path}/model_config.yaml"
    
    if not args.load_path is None:
        model_config_path = f"{args.load_path}/model_config.yaml"

    args = vars(args)
    
    args.pop("ds_short")
    args.pop("collagen")
    args.pop("seed")
    args.pop("ds_tag")
    args.pop("ds_base")
    args.pop("run_config_path")
    args.pop("model_config_path")

    if not args["continue_path"] is None:
        args["ds_path"] = None

    specified_args = {}
    # overwrite default args with args that are explicitly specified (default of flags must be False)
    for key in args.keys():
        if not args[key] is None:
            specified_args[key] = args[key]

    # tags for display and naming in evaluation data on several datasets
    if args["continue_path"] is None and not tags is None:
        specified_args["test_ds_tags"] = tags
        specified_args["description"] += tags

    if len(seeds)==1:
        run_from_config(run_config_path=run_config_path, model_config_path=model_config_path, idx=None, seed=seeds[0], **specified_args)
    else:
        for idx, seed in enumerate(seeds):
            run_from_config(run_config_path=run_config_path, model_config_path=model_config_path, idx=idx, seed=seed, **specified_args)

if __name__ == "__main__":
    run_client()