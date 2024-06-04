from grappa.training.experiment import Experiment
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

@hydra.main(version_base=None, config_path=str(Path(__file__).parent/"../configs"), config_name="train-default")
def main(cfg: DictConfig) -> None:

    # Loads the config of the pretrained model and overwrites the current config.
    if cfg.experiment.ckpt_path is not None and cfg.experiment.ckpt_cfg_override:
        ckpt_cfg_path = Path(cfg.experiment.warm_start).parent / 'config.yaml'
        ckpt_cfg = OmegaConf.load(ckpt_cfg_path)
        cfg.model = ckpt_cfg.model

    experiment = Experiment(config=cfg)
    experiment.train()
    experiment.test()


if __name__ == "__main__":
    main()
