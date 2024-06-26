from grappa.training.experiment import Experiment
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import logging


@hydra.main(version_base=None, config_path=str(Path(__file__).parent/"../configs"), config_name="evaluate")
def main(cfg: DictConfig) -> None:

    ckpt_path = Path(cfg.ckpt_path)

    assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist."
    
    if (ckpt_path.parent/'config.yaml').exists():
        ckpt_cfg = OmegaConf.load(ckpt_path.parent/'config.yaml')
    else:
        raise NotImplementedError(f"Checkpoint config not found at {ckpt_path.parent/'config.yaml'}")

    # add the test sets to the config:
    ckpt_cfg.data.data_module.pure_test_datasets += cfg.test_datasets
    ckpt_cfg.data.data_module.pure_test_datasets = list(set(ckpt_cfg.data.data_module.pure_test_datasets))

    # replace the checkpoint dir in ckpt_cfg with the one from the checkpoint path:
    ckpt_cfg.experiment.checkpointer.dirpath = ckpt_path.parent

    # determine the accelerator:
    ckpt_cfg.experiment.trainer.accelerator = cfg.accelerator
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Running on CPU.")
        ckpt_cfg.experiment.trainer.accelerator = 'cpu'

    experiment = Experiment(config=ckpt_cfg)
    experiment.test(ckpt_path=ckpt_path, n_bootstrap=cfg.n_bootstrap, test_data_path=cfg.test_data_path, load_split=True)
    experiment.eval_classical(ckpt_path=ckpt_path, classical_force_fields=cfg.classical_force_fields, test_data_path=cfg.test_data_path, load_split=True, n_bootstrap=cfg.n_bootstrap)


if __name__ == "__main__":
    main()
