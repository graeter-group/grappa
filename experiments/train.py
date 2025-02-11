from grappa.training import Experiment
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from grappa.training.experiment import use_tag_if_possible

@hydra.main(version_base=None, config_path=str(Path(__file__).parent/"../configs"), config_name="train")
def main(cfg: DictConfig) -> None:

    # loads a pretrained model from a checkpoint path or a tag.
    if cfg.experiment.ckpt_path is not None:
        ckpt_path = use_tag_if_possible(cfg.experiment.ckpt_path)
        ckpt_path = Path(ckpt_path)
        assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist."
        cfg.experiment.ckpt_path = str(ckpt_path)

    # Loads the config of the pretrained model and overwrites the current model config with that of the pretrained model.
    if cfg.experiment.ckpt_path is not None and cfg.experiment.ckpt_cfg_override:
        ckpt_cfg_path = Path(cfg.experiment.warm_start).parent / 'config.yaml'
        ckpt_cfg = OmegaConf.load(ckpt_cfg_path)
        cfg.model = ckpt_cfg.model

    experiment = Experiment(config=cfg, is_train=True)
    experiment.train()
    experiment.test(n_bootstrap=cfg.experiment.evaluation.n_bootstrap, load_split=False)


if __name__ == "__main__":
    main()
