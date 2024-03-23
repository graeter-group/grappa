#%%
from grappa.utils.loading_utils import model_dict_from_tag
import json
from pathlib import Path
#%%
# published grappa models have not only the state-dict of the trained model, but also the results of the training, the configuration used for training and the partition of the dataset into train, validation and test molecules.

# data_dict = model_dict_from_tag('grappa-1.1.0')['results']

data_dict = json.load(open('results.json'))

modified_dict = {}

#%%

# create a dictionary containing a the forcefields, Grappa-Amber99 and Grappa-AM1-BCC

for set_type in data_dict.keys():
    modified_dict[set_type] = {}

    for dataset in data_dict[set_type].keys():
        grappa_data = data_dict[set_type][dataset]['grappa']

        other_data = {ff: data_dict[set_type][dataset][ff] for ff in data_dict[set_type][dataset].keys() if 'grappa' not in ff}

        if '_amber99sbildn' in dataset or 'rad' in dataset:
            dataset = dataset.replace('_amber99sbildn', '')
            other_data['Grappa-ff99SB'] = grappa_data
        else:
            other_data['Grappa-AM1-BCC'] = grappa_data

        if dataset not in modified_dict[set_type]:
            modified_dict[set_type][dataset] = other_data
        else:
            modified_dict[set_type][dataset].update(other_data)
# %%
with open('results.json', 'w') as f:
    json.dump(modified_dict, f, indent=4)
# %%
