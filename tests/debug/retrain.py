from grappa.training.trainrun import do_trainrun
import yaml

project='tests'

config = yaml.load(open('grappa_config.yaml', 'r'), Loader=yaml.FullLoader)

do_trainrun(config=config, project=project)