#%%
from grappa.utils.loading_utils import model_dict_from_tag
import json

#%%
# published grappa models have not only the state-dict of the model, but also the results of the training, the configuration used for training and the partition of the dataset into train, validation and test molecules.

model_dict = model_dict_from_tag('grappa-1.1.0')

results = model_dict['results']
config = model_dict['config']
split_ids = model_dict['split_names']


with open('grappa-1.1.0_results.json', 'w') as f:
    json.dump(results, f, indent=4)
# %%
