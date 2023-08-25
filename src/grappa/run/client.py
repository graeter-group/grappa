
import argparse
from pathlib import Path
from grappa.run.run import run_from_config, get_default_run_config
from grappa.models.deploy import get_default_model_config
from grappa.constants import DEFAULTBASEPATH, DS_PATHS
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
    parser.add_argument('--seed','-s', type=int, default=None, help="random seed for the split of the dataset into train, val, test (default: 0)")
    parser.add_argument('--pretrain_steps', type=float, default=None, help="approximate number of gradient updates for pretraining (default: 2000)")
    parser.add_argument('--train_steps', type=float, default=None, help="approximate max number of gradient updates for training (default: 1e6)")
    parser.add_argument('--patience', type=float, default=None, help="patience of the learning rate scheduler in optimization steps (default: 2e3)")
    parser.add_argument('--plots', action='store_true', dest="plots", default=False, help="create plots during and at the end of training. might take time and memory (default: False)")
    parser.add_argument('--lr', type=float, default=None, help="the learning rate (does not apply to pretraining on parameters) (default: '1e-4')")
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--ds_short', default=None, type=str, nargs="+", help="codes for a collections of datasets that are added to the ds_paths. available options: \n'eric_nat': with amber charges, energies filtered at 60kcal/mol, \n'eric_rad': with amber charges (heavy), energies filtered at 60kcal/mol, \n'spice': with energies filtered at 60kcal/mol and filtered for standard amino acids, \n'eric' both of the above (default: [])")
    parser.add_argument('--scale_dict', default={'n4_improper_k':0.}, type=json.loads, help="dictionary of scaling factors for the different parameters in the direct-parameter-loss. Only has an effect if param_weight is nonzero. For every parameter that is not in the dictionary, 1. is assumed. input must be of the form '{\"n3_k\": 0.1, ...}'.(default: {'n4_improper_k':0., 'n3_eq':10., 'n3_k':10.})")
    parser.add_argument('--l2_dict', default=None, type=json.loads, help="dictionary of scaling factors for the different parameters in the direct-parameter-l2-regularisation. Every parameter that does not appear in the dictionary is not regularised. input must be of the form '{\"n3_k\": 0.1, ...}. (default: {})")
    parser.add_argument('--ds_split_names', default=None, type=str, help='Path to a file containing the names of the splits of the dataset. If None, the split is done according to the random seed. (default: None)')
    parser.add_argument('--time_limit', default=None, type=float, help='Time limit in hours. (default: 3)')


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
    parser.add_argument('--dense_layers_readout', type=int, default=None, help=" (default: 2)")
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
    parser.add_argument('--dropout', type=float, default=None, help=" (default: 0)")
    parser.add_argument('--rep_dropout', type=float, default=None, help=" (default: 0)")
    parser.add_argument('--final_dropout', dest='final_dropout', action='store_true', help='Whether to only apply one dropout layer for the representation, which is then located at the very end. The probability will be set to rep_droput. (default: False)')
    parser.add_argument('--no_final_dropout', dest='final_dropout', action='store_false')
    parser.set_defaults(final_dropout=None)
    parser.add_argument('--weight_decay', type=float, default=None, help=" (default: 0)")
    parser.add_argument('--attentional', dest='attentional', action='store_true')
    parser.add_argument('--no_attentional', dest='attentional', action='store_false')
    parser.set_defaults(attentional=None)

    parser.add_argument('--default_tag', type=str, default="med", help="A tag for the default parameters for the model. Can be ['small', 'med', 'large', 'deep'] (default: 'med')")
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

    if args.ds_path is None:
        args.ds_path = []

    tags = args.ds_tag
    if len(args.ds_tag)>0:
        args.ds_path += [str(Path(f"{args.ds_base}/{tag}")) for tag in args.ds_tag]

    # remove the tags in ds_short if they are in the list. then start again. (this emulates a goto for ds_short in [scan, pep, espaloma])
    while len(args.ds_short)>0:
        for ds_short in args.ds_short:

            if ds_short == "pep":
                args.ds_short += ["spice", "collagen", "radical_AAs", "radical_dipeptides"]

            elif ds_short == "espaloma":
                args.ds_short += ["spice_qca", "spice_monomers", "spice_pubchem"]
          
            elif ds_short == "scan":
                args.ds_short += ["scan_nat", "scan_rad"]

            elif ds_short == "opt":
                args.ds_short += ["opt_nat", "opt_rad"]

            elif ds_short in DS_PATHS.keys():
                args.ds_path.append(DS_PATHS[ds_short])
                tags.append(ds_short)

            else:
                raise ValueError(f"ds_short {ds_short} not recognized")

            args.ds_short.remove(ds_short)

    if args.ds_path is None:
        if args.run_config_path is None and args.continue_path is None:
            raise ValueError("either ds_path or ds_tag or a config path must be specified")



    if args.ds_tag == []:
        args.ds_tag = None

    if type(args.ds_path) == str:
        args.ds_path = [args.ds_path]

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

    run_from_config(run_config_path=run_config_path, model_config_path=model_config_path, idx=None, vpath=vpath, **specified_args)
    


def run_client():
    args = get_args()
    run_(args)


def full_run():
    args = get_args()
    vpath = []
    patience = copy.deepcopy(args.patience)
    param_weight = copy.deepcopy(args.param_weight)
    force_factor = copy.deepcopy(args.force_factor)
    energy_factor = copy.deepcopy(args.energy_factor)
    pretrain_steps = copy.deepcopy(args.pretrain_steps)

    if force_factor is None and energy_factor is None:
        args.force_factor = 10.0
        args.energy_factor = 1.0

    if patience is None:
        # args.patience = 1e4
        args.patience = 1e20

    if pretrain_steps is None:
        args.pretrain_steps = 1e4

    run_(args, vpath=vpath)
    
    continue_path = copy.deepcopy(vpath[0])
    assert not continue_path is None


    LRS = [1e-5, 1e-6]
    for lr in LRS:
        print("\nstarting run with lr ", lr)
        # loop over all arguments and set them to None if they are not a bool:
        for key, value in vars(args).items():
            if key in ["default_tag", "default_scale", "name", "ds_base"]:
                continue
            elif not type(value) == bool:
                setattr(args, key, None)
            
        
        args.continue_path = continue_path
        args.warmup = True
        args.lr = lr

        if args.time_limit is None:
            args.time_limit = get_default_run_config()["time_limit"]
        
        args.time_limit /= float(len(LRS))

        # other defaults for the next run:

        if patience is None:
            args.patience = 1e20
        if param_weight is None:
            args.param_weight = 1e-3

        if force_factor is None and energy_factor is None:
            args.force_factor = 1.0
            args.energy_factor = 1.0

        run_(args, vpath=[])

if __name__ == "__main__":
    run_client()