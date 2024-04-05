    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="ablation-grappa-1.0", help="Project name for wandb.")
parser.add_argument("--pretrain_path", type=str, default=None)
parser.add_argument("--with_hybridization", action="store_true", help="Use hybridization as input feature. Default is False.")
parser.add_argument("--wrong_symmetry", action="store_true", help="Use wrong symmetry of improper torsion. Default is False.")
parser.add_argument("--no_positional_encoding", action="store_true", help="Use no positional encoding. Default is False.")
parser.add_argument("--no_param_attention", action="store_true", help="Use no parameterized attention. Default is False.")
parser.add_argument("--no_gnn_attention", action="store_true", help="Use no gnn attention. Default is False.")
parser.add_argument("--no_gnn", action="store_true", help="Use no gnn at all. Default is False.")
parser.add_argument("--no_scaling", action="store_true", help="Use no scaling to approx. normally distributed nn output. Default is False.")
parser.add_argument("--exp_to_range", action="store_true", help="Use exp to range instead of shited elu. Default is False.")
parser.add_argument("--no_self_interaction", action="store_true", help="Use no self interaction. Default is False.")
parser.add_argument("--no_gated_torsion", action="store_true", help="Use no gated torsion. Default is False.")
parser.add_argument("--harmonic_gate", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":

    from grappa.training.trainrun import do_trainrun
    import yaml
    from pathlib import Path

    # load the config:
    config_path = "grappa_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # set the splitpath:    
    config["data_config"]["splitpath"] = str(Path(__file__).parent.parent.parent/f"dataset_creation/get_espaloma_split/espaloma_split.json")

    # set the name:
    config["trainer_config"]["name"] = ""

    if args.with_hybridization:
        config["model_config"]["in_feat_name"] += ["sp_hybridization"]
        config["trainer_config"]["name"] += "_hybrid"

    if args.pretrain_path is not None:
        config["lit_model_config"]['start_qm_epochs'] = 0


    if args.wrong_symmetry:
        config["model_config"]["worng_symmetry"] = True
        config["trainer_config"]["name"] += "_wrong_sym"

    if args.no_positional_encoding:
        config["model_config"]["positional_encoding"] = False
        config["trainer_config"]["name"] += "_no_pos_enc"    

    if args.no_param_attention:
        for lvl in ['bond', 'angle', 'proper', 'improper']:
            config["model_config"][f"{lvl}_symmetriser_depth"] += config["model_config"][f"{lvl}_transformer_depth"]
            config["model_config"][f"{lvl}_transformer_depth"] = 0

        config["trainer_config"]["name"] += "_no_param_att"

    if args.no_gnn_attention:
        config["model_config"]["gnn_convolutions"] += config["model_config"]["gnn_attentional_layers"]
        config["model_config"]["gnn_attentional_layers"] = 0
        config["trainer_config"]["name"] += "_no_gnn_att"

    if args.no_gnn:
        config["model_config"]["gnn_convolutions"] = 0
        config["model_config"]["gnn_attentional_layers"] = 0
        config["trainer_config"]["name"] += "_no_gnn"

    if args.no_self_interaction:
        config["model_config"]["self_interaction"] = False
        config["trainer_config"]["name"] += "_no_self_int"

    if args.no_gated_torsion:
        config["model_config"]["gated_torsion"] = False
        config["trainer_config"]["name"] += "_no_gated_torsion"

    if args.wrong_symmetry:
        config["model_config"]["wrong_symmetry"] = True
        config["trainer_config"]["name"] += "_wrong_sym"

    if args.harmonic_gate:
        config["model_config"]["harmonic_gate"] = True
        config["trainer_config"]["name"] += "_harmonic_gate"

    if args.no_scaling:
        raise NotImplementedError("No scaling not implemented yet.")
    
    if args.exp_to_range:
        raise NotImplementedError("Exp to range not implemented yet.")

    config['lit_model_config']['time_limit'] = 23.5 * 2

    # train:
    do_trainrun(config=config, project=args.project, pretrain_path=args.pretrain_path)