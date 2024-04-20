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
from pathlib import Path
from grappa.utils.loading_utils import model_dict_from_tag, model_from_dict
#%%

# load a dictionary specifying the model and its training configuration
model_dict = model_dict_from_tag('grappa-1.2')

# load the model from the dictionary
model = model_from_dict(model_dict)

# append an energy calculator
model = torch.nn.Sequential(model, Energy())

# load a dataset
DSNAME = 'uncapped_amber99sbildn'
dataset = Dataset.from_tag(DSNAME)


# split the dataset into train, validation and test sets according to the identifiers stored in the model dict:
split_ids = model_dict['split_names']
train_set, test_set, val_set = dataset.split(train_ids=split_ids['train'], val_ids=split_ids['val'], test_ids=split_ids['test'])


#%%
##############################################
# evaluate the model on the test set
##############################################
def evaluate(dataset, model):
    Path('figs').mkdir(exist_ok=True, parents=True)
    # create an evaluator that stores the results for each molecule:
    evaluator = Evaluator(keep_data=True, plot_dir='figs')

    # set dropout to 0, etc:
    model.eval().to('cuda' if torch.cuda.is_available() else 'cpu')

    # create a dataloader for batching:
    dataloader = GraphDataLoader(dataset, conf_strategy='max', batch_size=10, shuffle=False)

    for batch_idx, (graphs, ds_names) in enumerate(dataloader):

        print(f"batch {batch_idx}/{len(dataloader)-1}", end=None if batch_idx%5==0 or batch_idx==len(dataloader)-1 else '\r')

        with torch.no_grad():
            # forward pass
            graphs = model(graphs.to('cuda' if torch.cuda.is_available() else 'cpu'))

            # store the results
            evaluator.step(graphs.to('cpu'), ds_names)

    # print the results
    return evaluator

#%%
evaluator = evaluate(test_set, model)
results = evaluator.pool()
gradients = evaluator.all_gradients[DSNAME]
gradients_ref = evaluator.all_reference_gradients[DSNAME]
print(json.dumps(results, indent=4))
#%%
evaluator.plot_parameters(log=False, scatter=True, density=False)

import seaborn as sns
from matplotlib.colors import LogNorm

# randomly sample 10000 gradients and plot them


idxs = np.random.choice(gradients.shape[0], 5000, replace=False)
plt.scatter(gradients.numpy().flatten()[idxs], gradients_ref.numpy().flatten()[idxs], alpha=1, s=1, color='blue')

# sns.kdeplot(x=gradients.numpy().flatten()[idxs], y=gradients_ref.numpy().flatten()[idxs], cmap='Blues', fill=True, gridsize=50, thresh=0.00001, levels=8, norm=LogNorm(), alpha=0.5)

# %%
