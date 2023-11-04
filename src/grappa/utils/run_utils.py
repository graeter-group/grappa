import torch
from pathlib import Path

def load_weights_torchhub(url:str, filename:str) -> dict:
    models_path = Path(__file__).parents[3] / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    #torch.hub.set_dir('models_path')   # probably not necessary
    state_dict = torch.hub.load_state_dict_from_url(url, model_dir=str(models_path),file_name=filename)
    return state_dict