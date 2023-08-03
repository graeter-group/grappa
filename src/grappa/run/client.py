
import argparse
from pathlib import Path
from grappa.run.run import run_from_config
from grappa.models.deploy import get_default_model_config
from grappa.constants import DEFAULTBASEPATH
import copy
import json

def get_args():

    parser = argparse.ArgumentParser(
        epilog="If not using the ds_path argument, your dataset must be stored as PDBDataset in a folder named '{ds_base}/{ds_tag}'.",
        description="Run a model from a config file. If no config file is specified, a default config file is created.")

    parser.add_argument('--run_config_path', type=str, default=None, help="path to a config file, used for default arguments. if not specified or no config file is found, creates a default config file. (default: None)")
    parser.add_argument('--model_config_path', type=str, default=None, help="path to a config file, used for default arguments. if not specified or no config file is found, creates a default config file. (default: None)")
    parser.add_argument('--ds_path', type=str, nargs='+', default=None, help="path to dataset. setting this overwrites the effect of ds_tag. (default: None)")
    parser.add_argument('--ds_base', type=str, default=DEFAULTBASEPATH, help=f"if tags are used, this is the base path to the datasets (default: {DEFAULTBASEPATH})")
    parser.add_argument('-t', '--ds_tag', type=str, nargs='+', default=None, help=" (dataset must be stored as in files named '{ds_base}/{ds_tag}'. default: [])")
    parser.add_argument('--force_factor','-f', type=float, default=None, help=" (default: 1.)")
    parser.add_argument('--energy_factor','-e', type=float, default=None, help=" (default: 1.)")
    parser.add_argument('--recover_optimizer', action='store_true', default=False, help=" (if true, instead of a warm restart, we use the optimizer from the version specified by --load_path. default: False)")
    parser.add_argument('--warmup', action='store_true', default=False, help=" (if true, instead of a warm restart, we use the optimizer from the version specified by --load_path. default: False)")
    parser.add_argument('--continue_path', '-c', type=str, default=None, help="the version path of the version where we wish to continue training. the old logfile and models are stored, the rest is overwritten. (default: None)") 
    parser.add_argument('--param_weight', '-p', type=float, default=None, help=" (default: 0)")
    parser.add_argument('-d', '--description', type=str, nargs='+', default=[""], help='does nothing. only written in the config file for keeping track of several options that could be specified by hand like different datasets. (default: [""])')
    parser.add_argument('--load_path', '-l', type=str, default=None, help="path to the version-directory of the model to continue from. this overwrites the pretraining options, i.e. if both are specified, there is no pretraining. either absolute or relative to the path from where the command is being run. (default: None)")
    parser.add_argument('--confs', type=int, default=None, help=" (default: None)")
    parser.add_argument('--mols', type=int, default=None, help=" (default: None)")
    parser.add_argument('--storage_path', type=str, default=None, help="The path in which the version-directories are stored. (default: 'versions')")
    parser.add_argument('--name', type=str, default='', help="Optional: The name of the version directory (up to indices). (default: '')")
    parser.add_argument('--n', type=int, default=None, help="how many times to run (default: None)") # NOTE: this might be unnecesary now that we have several seeds
    parser.add_argument('--test', action='store_true', default=False, help="Reducing dataset size to 50 graphs for testing functionality. (default: False)'")
    parser.add_argument('--seed','-s', type=int, nargs='+', default=None, help="random seed for the split of the dataset into train, val, test (default: 0)")
    parser.add_argument('--pretrain_steps', type=float, default=None, help="approximate number of gradient updates for pretraining (default: 500)")
    parser.add_argument('--train_steps', type=float, default=None, help="approximate max number of gradient updates for training (default: 100000.0)")
    parser.add_argument('--patience', type=float, default=None, help="ratio of maximum total steps and patience of the learning rate scheduler, i.e. if (default: 0.0001)")
    parser.add_argument('--plots', action='store_true', dest="plots", default=False, help="create plots during and at the end of training. might take time and memory (default: False)")
    parser.add_argument('--lr', type=float, default=None, help="the learning rate (does not apply to pretraining on parameters) (default: '1e-6')")
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--ds_short', default=None, type=str, nargs="+", help="codes for a collections of datasets that are added to the ds_paths. available options: \n'eric_nat': with amber charges, energies filtered at 60kcal/mol, \n'eric_rad': with amber charges (heavy), energies filtered at 60kcal/mol, \n'spice': with energies filtered at 60kcal/mol and filtered for standard amino acids, \n'eric' both of the above (default: [])")
    parser.add_argument('--collagen', default=False, action="store_true", help="Whether or not to use the collagen forcefield and dataset, i.e. including hyp and dop. (default: False)")
    parser.add_argument('--scale_dict', default=None, type=json.loads, help="dictionary of scaling factors for the different parameters in the direct-parameter-loss. Only has an effect if param_weight is nonzero. For every parameter that is not in the dictionary, 1. is assumed. input must be of the form '{\"n3_k\": 0.1, ...}.(default: {'n4_improper_k':0., 'n3_eq':10., 'n3_k':10.})")
    parser.add_argument('--l2_dict', default=None, type=json.loads, help="dictionary of scaling factors for the different parameters in the direct-parameter-l2-regularisation. Every parameter that does not appear in the dictionary is not regularised. input must be of the form '{\"n3_k\": 0.1, ...}. (default: {'n4_k':1., 'n4_improper_k': 10.})")


    parser.add_argument('--n_conv', type=int, default=None, help=" (default: 3)")
    parser.add_argument('--n_att', type=int, default=None, help=" (default: 3)")
    parser.add_argument('--n_heads', type=int, default=None, help="Number of attention heads in the graph attention model. (default: 6)")
    parser.add_argument('--width', type=int, default=None, help="of the representation network (default: 512)")
    parser.add_argument('--rep_feats', type=int, default=None, help=" (default: 512)")
    parser.add_argument('--readout_width', type=int, default=None, help=" (default: 512)")
    parser.add_argument('--in_feat_name', type=str, nargs='+', default=None, help='which input features the model should use. Can be ["atomic_number", "in_ring", "q_ref", "is_radical", "degree", "residue", "mass", "additional_features"] (default: ["atomic_number", "in_ring", "q_ref", "is_radical"])')
    parser.add_argument('--improper', dest='use_improper', action='store_true')
    parser.add_argument('--no_improper', dest='use_improper', action='store_false')
    parser.set_defaults(use_improper=None)
    parser.add_argument('--old_model', dest='old_model', action='store_true')
    parser.add_argument('--no_old_model', dest='old_model', action='store_false')
    parser.set_defaults(old_model=None)
    parser.add_argument('--add_feat', '-a', type=str, nargs='+', default=None, help='Features that are added to default features if not in there already (shortcut for if one wishes to add to exsiting defaults) Can be ["atomic_number", "in_ring", "q_ref", "is_radical", "degree", "residue", "mass", "additional_features"]. (default: None)')
    parser.add_argument('--dense_layers', type=int, default=None, help=" (default: 2)")
    parser.add_argument('--n_att_readout', type=int, default=None, help=" (default: 2)")
    parser.add_argument('--n_heads_readout', type=int, default=None, help=" (default: 8)")
    parser.add_argument('--reducer_feats', type=int, default=None, help=" (default: None)")
    parser.add_argument('--attention_hidden_feats', type=int, default=None, help=" (default: None)")
    parser.add_argument('--positional_encoding', dest='positional_encoding', action='store_true')
    parser.add_argument('--no_positional_encoding', dest='positional_encoding', action='store_false')
    parser.set_defaults(positional_encoding=None)
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true')
    parser.add_argument('--no_layer_norm', dest='layer_norm', action='store_false')
    parser.set_defaults(layer_norm=None)
    parser.add_argument('--dropout', type=float, default=None, help=" (default: 0.2)")
    parser.add_argument('--rep_dropout', type=float, default=None, help=" (default: 0)")
    parser.add_argument('--weight_decay', type=float, default=None, help=" (default: 1e-4)")
    parser.add_argument('--attentional', dest='attentional', action='store_true')
    parser.add_argument('--no_attentional', dest='attentional', action='store_false')
    parser.set_defaults(attentional=None)

    parser.add_argument('--default_tag', type=str, default="small", help="A tag for the default parameters for the model. Can be ['small', 'med', 'large'] (default: 'small')")
    parser.add_argument('--default_scale', type=float, default=1, help="A scale factor for the default parameters for the model.  Only affects widths.")

    args = parser.parse_args()
    return args


def run_(args, vpath=[]):

    args = copy.deepcopy(args)

    if not args.add_feat is None:
        if args.in_feat_name is None:
            args.in_feat_name = get_default_model_config(args.default_tag, scale=args.default_scale)["in_feat_name"]
        for f in args.add_feat:
            if not f in args.in_feat_name:
                args.in_feat_name.append(f)

    # loop over args and set to None whenever the argument is the string 'None'
    for key, value in vars(args).items():
        if value == "None" or value == ["None"]:
            setattr(args, key, None)

    if args.ds_short is None:
        args.ds_short = []
    if args.ds_tag is None:
        args.ds_tag = []

    for ds_short in args.ds_short:
        suffix = "_filtered"
        suffix_col = ""
        if args.collagen:
            suffix_col = "_col"

        if ds_short == "eric_nat":
            args.ds_tag += [f'AA_scan_nat/charge_default{suffix_col}_ff_amber99sbildn{suffix}', f'AA_opt_nat/charge_default{suffix_col}_ff_amber99sbildn{suffix}']

        elif ds_short == "eric_rad":
            args.ds_tag += [f'AA_scan_rad/charge_heavy{suffix_col}_ff_amber99sbildn{suffix}', f'AA_opt_rad/charge_heavy{suffix_col}_ff_amber99sbildn{suffix}']

        elif ds_short == "spice":
            args.ds_tag += [f'spice/charge_default_ff_amber99sbildn{suffix}']

        elif ds_short == "spice_openff":
            args.ds_tag += [f'spice_openff/charge_default_ff_gaff-2_11{suffix}']

        elif ds_short == "spice_qca":
            args.ds_tag += [f'qca_spice/charge_default_ff_gaff-2_11{suffix}']

        elif ds_short == "spice_monomers":
            args.ds_tag += [f'monomers/charge_default_ff_gaff-2_11{suffix}']

        elif ds_short == "eric":
            args.ds_short.remove("eric")
            args.ds_short += [f'AA_scan_nat/charge_default{suffix_col}_ff_amber99sbildn{suffix}', f'AA_opt_nat/charge_default{suffix_col}_ff_amber99sbildn{suffix}']
            args.ds_short += [f'AA_scan_rad/charge_heavy{suffix_col}_ff_amber99sbildn{suffix}', f'AA_opt_rad/charge_heavy{suffix_col}_ff_amber99sbildn{suffix}']
        
        else:
            raise ValueError(f"ds_short {ds_short} not recognized")


    if len(args.ds_tag)>0:
        if args.ds_path is None:
            args.ds_path = []
        args.ds_path += [str(Path(f"{args.ds_base}/{tag}")) for tag in args.ds_tag]


    tags = None
    if args.ds_path is None:
        if args.run_config_path is None and args.continue_path is None:
            raise ValueError("either ds_path or ds_tag or a config path must be specified")


    

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
        run_config_path = f"{args.continue_path}/run_config.yml"

    if not args.continue_path is None:
        model_config_path = f"{args.continue_path}/model_config.yml"
    
    if not args.load_path is None:
        model_config_path = f"{args.load_path}/model_config.yml"

    args = vars(args)
    
    args.pop("ds_short")
    args.pop("collagen")
    args.pop("seed")
    args.pop("ds_tag")
    args.pop("ds_base")
    args.pop("run_config_path")
    args.pop("model_config_path")
    args.pop("add_feat")

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
        run_from_config(run_config_path=run_config_path, model_config_path=model_config_path, idx=None, seed=seeds[0], vpath=vpath, **specified_args)
    else:
        for idx, seed in enumerate(seeds):
            run_from_config(run_config_path=run_config_path, model_config_path=model_config_path, idx=idx, seed=seed, vpath=vpath, **specified_args)


def run_client():
    args = get_args()
    run_(args)


def full_run():
    args = get_args()
    vpath = []
    run_(args, vpath=vpath)
    
    continue_path = vpath[0]
    assert not continue_path is None

    warmup = True
    lr = 1e-6
    
    # loop over all arguments and set them to None if they are not a bool:
    for key, value in vars(args).items():
        if key in ["default_tag", "default_scale", "name", "ds_base"]:
            continue
        elif not type(value) == bool:
            setattr(args, key, None)
        
    
    args.continue_path = continue_path
    args.warmup = warmup
    args.lr = lr

    run_(args)


if __name__ == "__main__":
    run_client()