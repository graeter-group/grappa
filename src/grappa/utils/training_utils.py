import torch
from grappa.models import GrappaModel
import yaml
from pathlib import Path
import pandas as pd
import logging


def get_model_from_checkpoint(ckpt: dict, model_config:dict, device: torch.device='cpu') -> GrappaModel:
    model = GrappaModel(**model_config).to(device)
    state_dict = ckpt['state_dict']
    # grappa model wieghts are stored as model.0 (zeroth entry in torch.sequential and model entry of ckpt state dict):
    state_dict = {k.replace('model.0.', ''): v for k, v in state_dict.items() if 'model.0.' in k}
    model.load_state_dict(state_dict=state_dict)
    return model

def get_model_from_path(ckpt_path: str, device: torch.device='cpu') -> GrappaModel:
    logging.info(f"Loading model from {str(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location=device)
    with open(Path(ckpt_path).parent / 'config.yaml', 'r') as f:
        model_config = yaml.safe_load(f)['model']
    return get_model_from_checkpoint(ckpt, model_config, device)


def to_df(summary: dict, short:bool=False) -> pd.DataFrame:
    df = pd.DataFrame()
    for dsname, metrics in summary.items():
        for metric in (['n_mols', 'n_confs', 'rmse_energies', 'crmse_gradients'] if short else metrics.keys()):
            value = metrics[metric]
            if isinstance(value, dict):
                df.loc[dsname, metric] = f"{value['mean']:.2f}+-{value['std']:.2f}"
            else:
                if metric in ['n_mols', 'n_confs']:
                    df.loc[dsname, metric] = f"{value:3d}"
                else:
                    df.loc[dsname, metric] = f"{value:.2f}"

    return df