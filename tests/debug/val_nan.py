import yaml
from grappa.training.resume_trainrun import resume_trainrun
from grappa.training.trainrun import safe_trainrun
from grappa.utils.dataset_utils import get_data_path


# run_id = 'p1e1ojyd'
run_id = 'lembf22v'

project = 'hpo_grappa_final'
project = 'tests'

# wandb_folder = '/hits/fast/mbm/seutelf/grappa/experiments/hyperparameter_optimization/wandb'
wandb_folder = 'wandb'




overwrite_config = {
    # 'data_config': {
    #     "datasets": [
    #         str(get_data_path()/"dgl_datasets"/dsname) for dsname in
    #         [
    #             "rna-diverse",
    #             'spice-des-monomers',
    #         ]
    #     ],
    #     'splitpath': None,               
    # }
}

resume_trainrun(run_id, project, wandb_folder, new_wandb_run=True, overwrite_config=overwrite_config)

