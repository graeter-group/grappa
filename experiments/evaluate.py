from grappa.training.experiment import Experiment, use_tag_if_possible
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import copy
import torch
import logging


@hydra.main(version_base=None, config_path=str(Path(__file__).parent/"../configs"), config_name="evaluate")
def main(cfg: DictConfig) -> None:

    ckpt_path = use_tag_if_possible(cfg.evaluate.ckpt_path)
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist."
    
    if (ckpt_path.parent/'config.yaml').exists():
        ckpt_cfg = OmegaConf.load(ckpt_path.parent/'config.yaml')
    else:
        raise NotImplementedError(f"Checkpoint config not found at {ckpt_path.parent/'config.yaml'}")

    ckpt_data_config = copy.deepcopy(ckpt_cfg.data)
    # overwrite the data config (we use the split from the checkpoint to differentiate between training and test data):
    if cfg.evaluate.datasets is not None:
        ckpt_cfg.data.data_module.datasets = cfg.evaluate.datasets
    if cfg.evaluate.pure_test_datasets is not None:
        ckpt_cfg.data.data_module.pure_test_datasets = cfg.evaluate.pure_test_datasets
    ckpt_cfg.data.data_module.pure_train_datasets = []
    ckpt_cfg.data.data_module.pure_val_datasets = []

    # replace the checkpoint dir in ckpt_cfg with the one from the checkpoint path:
    ckpt_cfg.experiment.checkpointer.dirpath = ckpt_path.parent

    # determine the accelerator:
    ckpt_cfg.experiment.trainer.accelerator = cfg.evaluate.accelerator
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Running on CPU.")
        ckpt_cfg.experiment.trainer.accelerator = 'cpu'

    # setting some default args:
    OmegaConf.set_struct(cfg, False)
    if not hasattr(cfg.evaluate,'classical_force_fields') or cfg.evaluate.classical_force_fields is None:
        cfg.evaluate.classical_force_fields = []
    if not hasattr(cfg.evaluate,'gradient_contributions') or cfg.evaluate.gradient_contributions is None:
        cfg.evaluate.gradient_contributions = []
    if not hasattr(cfg.evaluate,'compare_forcefields') or cfg.evaluate.compare_forcefields is None:
        cfg.evaluate.compare_forcefields = []
    if not hasattr(cfg.evaluate,'splitpath'):
        cfg.evaluate.splitpath = None

    OmegaConf.set_struct(cfg, True)
    if hasattr(cfg.evaluate,'ff_lookup') and cfg.evaluate.ff_lookup is not None and len(cfg.evaluate.ff_lookup.keys()) > 0:
        # load the forcefield lookup:
        ckpt_cfg.data.data_module.ff_lookup.update(cfg.evaluate.ff_lookup)

    # if specified, only predict certain parameters with grappa (overwrite in data config):
    if hasattr(cfg.evaluate,'grappa_contributions') and cfg.evaluate.grappa_contributions is not None:
        complement = list(set(['nonbonded', 'bond', 'angle', 'proper', 'improper']) - set(cfg.evaluate.grappa_contributions))
        ckpt_cfg.data.data_module.ref_terms = complement
        ckpt_cfg.data.energy.terms = cfg.evaluate.grappa_contributions

    experiment = Experiment(config=ckpt_cfg, load_data=False)

    if cfg.evaluate.eval_model:
        experiment.test(ckpt_path=ckpt_path, n_bootstrap=cfg.evaluate.n_bootstrap, test_data_path=cfg.evaluate.test_data_path, load_split=True, plot=cfg.evaluate.plot, gradient_contributions=cfg.evaluate.gradient_contributions, ckpt_data_config=ckpt_data_config, store_test_data=cfg.evaluate.store_test_data, splitpath=cfg.evaluate.splitpath)

    experiment.eval_classical(ckpt_path=ckpt_path, classical_force_fields=cfg.evaluate.classical_force_fields, test_data_path=cfg.evaluate.test_data_path, load_split=cfg.evaluate.eval_model, n_bootstrap=cfg.evaluate.n_bootstrap, plot=cfg.evaluate.plot, gradient_contributions=cfg.evaluate.gradient_contributions, store_test_data=cfg.evaluate.store_test_data, splitpath=cfg.evaluate.splitpath)
    experiment.compare_forcefields(ckpt_path=ckpt_path, forcefields=cfg.evaluate.compare_forcefields, gradient_contributions=cfg.evaluate.gradient_contributions)

if __name__ == "__main__":
    main()
