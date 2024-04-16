#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from grappa.data.Dataset import Dataset
from grappa.data.GraphDataLoader import GraphDataLoader
from grappa.training.evaluation import Evaluator
from grappa.models.energy import Energy
from grappa.utils import loading_utils
from grappa.utils.loading_utils import model_dict_from_tag, model_from_dict
#%%

# load a dictionary specifying the model and its training configuration
model_dict = model_dict_from_tag('grappa-1.1.0')

# load the model from the dictionary
model = model_from_dict(model_dict)

# append an energy calculator
model = torch.nn.Sequential(model, Energy())

# load a dataset
dataset = Dataset.from_tag('spice-dipeptide')


# split the dataset into train, validation and test sets according to the identifiers stored in the model dict:
split_ids = model_dict['split_names']
train_set, test_set, val_set = dataset.split(train_ids=split_ids['train'], val_ids=split_ids['val'], test_ids=split_ids['test'])


#%%
##############################################
# evaluate the model on the test set
##############################################
def evaluate(dataset, model):
    # create an evaluator that stores the results for each molecule:
    evaluator = Evaluator(keep_data=True)

    # set dropout to 0, etc:
    model.eval()

    # create a dataloader for batching:
    dataloader = GraphDataLoader(dataset, conf_strategy='max', batch_size=10, shuffle=False)

    for batch_idx, (graphs, ds_names) in enumerate(dataloader):

        print(f"batch {batch_idx}/{len(dataloader)-1}", end=None if batch_idx%5==0 or batch_idx==len(dataloader)-1 else '\r')

        with torch.no_grad():
            # forward pass
            graphs = model(graphs)

            # store the results
            evaluator.step(graphs, ds_names)

    # print the results
    return evaluator

#%%
evaluator = evaluate(test_set, model)
results = evaluator.pool()
gradients = evaluator.all_gradients['spice-dipeptide']
gradients_ref = evaluator.all_reference_gradients['spice-dipeptide']
print(json.dumps(results, indent=4))

plt.scatter(gradients_ref, gradients)
# %%
