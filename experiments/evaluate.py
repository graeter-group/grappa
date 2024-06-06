from grappa.training.experiment import Experiment
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

@hydra.main(version_base=None, config_path=str(Path(__file__).parent/"../configs"), config_name="evaluate")
def main(cfg: DictConfig) -> None:

    ckpt_path = Path(cfg.ckpt_path)

    assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist."
    
    if (ckpt_path.parent/'config.yaml').exists():
        ckpt_cfg = OmegaConf.load(ckpt_path.parent/'config.yaml')
    else:
        raise NotImplementedError(f"Checkpoint config not found at {ckpt_path.parent/'config.yaml'}")

    experiment = Experiment(config=ckpt_cfg)
    experiment.test(ckpt_path=ckpt_path, n_bootstrap=cfg.n_bootstrap, store_data=cfg.test_data_path)


if __name__ == "__main__":
    main()
