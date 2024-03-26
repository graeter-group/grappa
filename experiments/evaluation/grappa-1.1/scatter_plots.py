#%%
from grappa.utils.loading_utils import model_dict_from_tag, model_from_tag
from grappa.training.evaluation import Evaluator
from grappa.training.get_dataloaders import get_dataloaders

import numpy as np, torch, matplotlib.pyplot as plt
from collections import defaultdict
from grappa.utils.dgl_utils import unbatch

from grappa.models.energy import Energy
from typing import Dict, List

#%%
DEVICE = 'cpu'

# Load the model
model = model_from_tag('grappa-1.1.0')
model = torch.nn.Sequential(model, Energy())

split_names = model_dict_from_tag('grappa-1.1.0')['split_names']
# %%

# Load the data
datasets = [
    'spice-des-monomers',
    'spice-pubchem',
    'spice-dipeptide',
    'rna-trinucleotide',
]
_,_,test_loader = get_dataloaders(datasets=datasets, split_ids=split_names, test_batch_size=1, keep_features=True)
#%%
g, dsnames = next(iter(test_loader))
g = model(g)
g.nodes['n1'].data['atomic_number'].argmax(dim=1) + 1
# %%
def get_data(model, loader, compare_ff)->Dict[str, Dict[str, List[np.ndarray]]]:
    '''
    Returns {dsname: grappa_energies, grappa_gradients, compare_ff_energies, compare_ff_gradients, qm_energies, qm_gradients, grappa_node_encodings, elements}
    
    all of these are dictionaries with the dataset name as key and the values as a list of numpy arrays.

    The grappa and ref energies contain the nonbonded contribution, for grappa this means we add the difference between qm and ..._ref to grappas prediction, for the compare_ff we take the full contribution.
    '''
    data = defaultdict(list)

    model = model.to(DEVICE)

    if type(compare_ff) == str:
        compare_ff = [compare_ff]

    for i, (g, dsnames) in enumerate(loader):
        print(f'batch {i+1}/{len(loader)}', end='\r')
        if i > 2 and i < 150:
            continue
        g = g.to(DEVICE)
        g = model(g)
        g = g.cpu()

        graphs = unbatch(g)

        for i, (graph, dsname) in enumerate(zip(graphs, dsnames)):
            qm_energies = graph.nodes['g'].data['energy_qm'].detach().clone().numpy()
            qm_energies -= qm_energies.mean()
            qm_gradients = graph.nodes['n1'].data['gradient_qm'].detach().clone().numpy()

            nonbonded_energies = qm_energies - graph.nodes['g'].data['energy_ref'].detach().clone().numpy()
            nonbonded_gradients = qm_gradients - graph.nodes['n1'].data['gradient_ref'].detach().clone().numpy()

            grappa_energies = nonbonded_energies + graph.nodes['g'].data['energy'].detach().clone().numpy()
            grappa_energies -= grappa_energies.mean()
            grappa_gradients = nonbonded_gradients + graph.nodes['n1'].data['gradient'].detach().clone().numpy()

            elements = (g.nodes['n1'].data['atomic_number'].argmax(dim=1) + 1).detach().clone().numpy()
            grappa_encoding = g.nodes['n1'].data['h'].detach().clone().numpy()

            d = {
                'grappa_energies': grappa_energies.flatten(),
                'grappa_gradients': grappa_gradients.flatten(),
                'qm_energies': qm_energies.flatten(),
                'qm_gradients': qm_gradients.flatten(),
                'grappa_node_encodings': grappa_encoding,
                'elements': elements,
            }

            for ff in compare_ff:
                if f'energy_{ff}' not in graph.nodes['g'].data.keys():
                    continue
                compare_ff_energies = graph.nodes['g'].data[f'energy_{ff}'].detach().clone().numpy()
                compare_ff_energies -= compare_ff_energies.mean()
                compare_ff_gradients = graph.nodes['n1'].data[f'gradient_{ff}'].detach().clone().numpy()

                d[f'{ff}_energies'] = compare_ff_energies.flatten()
                d[f'{ff}_gradients'] = compare_ff_gradients.flatten()


            data[dsname].append(d)

    # reformat the dict such that it is dsname: {key: [values]}
    inner_keys = set()
    for dsname, dlist in data.items():
        for d in dlist:
            inner_keys.update(d.keys())

    reformatted_data = {}
    for dsname, d in data.items():
        reformatted_data[dsname] = {k: [v[k] for v in d if k in v] for k in inner_keys}

    return reformatted_data

# %%
data = get_data(model, test_loader, ['gaff-2.11', 'amber14'])
data.keys()
# %%
def scatter_plot(data, dataset, force:bool=True, ax=None, compare_ff='gaff-2.11', compare_ff_name='Gaff 2.11', label:str='name', symmetric:bool=True, no_yscale:bool=False, title:str=None, n_max:int=None, s=3, alpha=1, seed=0, legend:bool=True, nograppa:bool=False, fix_scale:bool=False, unit:bool=True, no_ticks:bool=False)->plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    FONTSIZE = 22
    FONT = 'Arial'

    GRAPPA_COLOR = '#1f77b4'
    AMBER_COLOR = '#e41a1c'

    S = s
    ALPHA = alpha

    plt.rc('font', family=FONT)
    plt.rc('xtick', labelsize=FONTSIZE)
    plt.rc('ytick', labelsize=FONTSIZE)
    plt.rc('axes', labelsize=FONTSIZE, titlesize=FONTSIZE+2)
    plt.rc('legend', fontsize=FONTSIZE, title_fontsize=FONTSIZE-2)


    d = data[dataset]
    if force:
        x = [-v for v in d['qm_gradients']]
        y = [-v for v in d['grappa_gradients']]
        y_compare = [-v for v in d[f'{compare_ff}_gradients']]
        unit_ = 'kcal/mol/Å'
        ax.set_xlabel(f'QM force [{unit_}]' if unit else 'QM force')
        if not no_yscale:            
            ax.set_ylabel(f'Prediction [{unit_}]' if unit else 'Prediction')
    else:
        x = d['qm_energies']
        y = d['grappa_energies']
        y_compare = d[f'{compare_ff}_energies']

        unit_ = 'kcal/mol'
        ax.set_xlabel(f'QM energy [{unit_}]' if unit else 'QM energy')
        if not no_yscale:            
            ax.set_ylabel(f'Prediction [{unit_}]' if unit else 'Prediction')

    if title is not None:
        ax.set_title(title, fontsize=FONTSIZE+2)

    # concat all the data
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    y_compare = np.concatenate(y_compare, axis=0)

    if not force:
        # shift all energies by the same value such that the min/max of QM energy is symmetric around zero.
        # energies have no absolute meaning, thus we can do this
        spread = (x.max() + x.min())/2
        x -= spread
        y -= spread
        y_compare -= spread

    if n_max is not None and n_max < len(x):
        np.random.seed(seed)
        idxs = np.random.choice(len(x), n_max, replace=False)
        x = x[idxs]
        y = y[idxs]
        y_compare = y_compare[idxs]

    def rmse(x, y):
        return np.sqrt(np.mean((x-y)**2))

    label_grappa = 'Grappa'
    label_compare = compare_ff_name
    if label == 'rmse':
        label_grappa = f'RMSE: {rmse(x, y):.2f}'# {unit}'
        label_compare = f'RMSE: {rmse(x, y_compare):.2f}'# {unit}'
    elif label == 'combined':
        label_grappa = f'{label_grappa} ({rmse(x, y):.2f})'
        label_compare = f'{label_compare} ({rmse(x, y_compare):.2f})'
    

    ax.scatter(x, y_compare, alpha=ALPHA, s=S, label=label_compare, color=AMBER_COLOR)
    ax.scatter(x, y, alpha=ALPHA, s=S, label=label_grappa, color=GRAPPA_COLOR)

    min_val = min([x.min(), y.min(), y_compare.min()])
    max_val = max([x.max(), y.max(), y_compare.max()])


    if symmetric:
        val = max(abs(min_val), abs(max_val))
        min_val = -val
        max_val = val

    if fix_scale:
        if not force:
            min_val = -45
            max_val = 45

        else:
            min_val = -99
            max_val = 99

    ax.set_ylim(min_val, max_val)
    ax.set_xlim(min_val, max_val)

    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='-', linewidth=1)

    if legend:
        # reverse the legend order and set the size of the legend markers
        handles, labels = ax.get_legend_handles_labels()
        if nograppa:
            handles = handles[:-1]
            labels = labels[:-1]
        # legend = ax.legend(handles[::-1], labels[::-1], loc='upper left', frameon=False, handlelength=0.6, handletextpad=0.05, borderaxespad=0.)
        legend = ax.legend(handles[::-1], labels[::-1], loc='upper left', frameon=False, handlelength=0.9, handletextpad=0.1, borderaxespad=0.)

        for handle in legend.legend_handles:
            handle.set_sizes([50])  # Set to desired size
            # set alpha:
            handle.set_alpha(1)


    num_ticks = 5
    ax.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
    ax.yaxis.set_major_locator(plt.MaxNLocator(num_ticks))

    # make the ticks go into and out of the plot and make them larger:
    ax.tick_params(axis='both', which='major', direction='inout', length=10, width=1)

    if fix_scale:    
        # only use every second tick label:
        for label in ax.get_xticklabels()[::2]:
            label.set_visible(False)
        for label in ax.get_yticklabels()[::2]:
            label.set_visible(False)

    if no_ticks:
        # turn off tick labels:
        ax.set_yticklabels([])

    # re-set the limits
    ax.set_ylim(min_val, max_val)
    ax.set_xlim(min_val, max_val)

    return ax


# %%
scatter_plot(data, 'spice-dipeptide', force=True, label='name', compare_ff='amber14', compare_ff_name='AmberFF14SB', n_max=10000, s=3, alpha=1, nograppa=True)

#%%
DATASETS = [
    'spice-pubchem',
    'spice-dipeptide',
    'rna-trinucleotide',
]
DSNAMES = [
    'Small Molecules',
    'Dipeptides',
    # 'Trinucleotides',
    'RNA',
]
FFS = [
    'gaff-2.11',
    'amber14',
    'amber14'
]

FF_NAMES = [
    'Gaff 2.11',
    'FF14SB',
    'RNA.OL3'
]

figsize = 4.5
figsize=4
padding = 2

N=5000

s=2

with_forces=False

fig, axes = plt.subplots(1+int(with_forces), len(DATASETS), figsize=(len(DATASETS)*figsize, (float(with_forces)+1.)*figsize + 2*padding*0.05))
for i, dataset_name in enumerate(DATASETS):
    # axes[0, i] = scatter_plot(data, dataset_name, force=False, compare_ff=FFS[i], compare_ff_name=FF_NAMES[i], ax=axes[0, i], no_yscale=(i>0), title=DSNAMES[i], n_max=10000, nograppa=True, fix_scale=True, unit=False, no_ticks=(i>0))
    # axes[1, i] = scatter_plot(data, dataset_name, force=True, compare_ff=FFS[i], compare_ff_name=FF_NAMES[i], ax=axes[1, i], no_yscale=(i>0), n_max=10000, legend=False, nograppa=True, fix_scale=True, unit=False, no_ticks=(i>0))

    scatter_plot(data, dataset_name, force=False, compare_ff=FFS[i], compare_ff_name=FF_NAMES[i], ax=axes[0, i] if with_forces else axes[i], no_yscale=(i>0), title=DSNAMES[i], n_max=N, fix_scale=True, s=s)
    if with_forces:
        axes[1, i] = scatter_plot(data, dataset_name, force=True, compare_ff=FFS[i], compare_ff_name=FF_NAMES[i], ax=axes[1, i], no_yscale=(i>0), n_max=N, legend=False, fix_scale=True, s=s)

# tight layout
fig.tight_layout(h_pad=padding)
fig.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')
#%%


# %%
from sklearn.decomposition import PCA

seed = 0
np.random.seed(seed)

x = [data[dsname]['grappa_node_encodings'] for dsname in data.keys()]
# flatten list along zeroth axis
x = [item for sublist in x for item in sublist]

x = np.concatenate(x, axis=0).astype(float)

y = [data[dsname]['elements'] for dsname in data.keys()]
y = [item for sublist in y for item in sublist]
y = np.concatenate(y, axis=0).astype(int)



# for each element, randomly sample N points with replacement:
N = 2000
x_new = []
y_new = []
for elem in np.unique(y):
    idxs = np.where(y == elem)[0]
    idxs = np.random.choice(idxs, N, replace=True)
    x_new.append(x[idxs])
    y_new.append(y[idxs])

x = np.concatenate(x_new, axis=0)
y = np.concatenate(y_new, axis=0)


ELEMS = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    15: 'P',
    16: 'S',
    17: 'Cl',
    35: 'Br',
    53: 'I',
}

# Assume x is your N x feat_dim NumPy array
# x = np.random.rand(N, feat_dim)  # Example initialization

# Perform PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)

#%%

# Plot the first two principal components
fig, ax = plt.subplots(figsize=(5,5))

FONT = 'Arial'
FONTSIZE = 21

plt.rc('font', family=FONT)
plt.rc('xtick', labelsize=FONTSIZE)
plt.rc('ytick', labelsize=FONTSIZE)
plt.rc('axes', labelsize=FONTSIZE)
plt.rc('legend', fontsize=FONTSIZE+2)


for elem in range(100):
    if elem in y:
        idxs = y == elem
        ax.scatter(x_pca[idxs, 0], x_pca[idxs, 1], alpha=1, s=3, label=ELEMS[elem])
# ax.set_title('2D PCA of atom embeddings', fontsize=FONTSIZE)
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Place the legend outside the plot
# PLACE IT UNDERNEATH THE PLOT
legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=True, handlelength=0.8, handletextpad=0.3)

for handle in legend.legendHandles:
    handle.set_sizes([80])  # Set to desired size
    handle.set_alpha(1)  # Set alpha
    # handle.set_y(-5)  # Example of how to adjust positioning, commented out
name = 'pca.png'


# turn off ticks entirely:
ax.set_xticks([])
ax.set_yticks([])

ax.set_xlabel(r'$u_1$', fontsize=FONTSIZE-1)
ax.set_ylabel(r'$u_2$', fontsize=FONTSIZE-1)

ax.set_title('Learned Atom Embeddings', fontsize=FONTSIZE)

# save:

fig.savefig(name, dpi=300, bbox_inches='tight')

ax

# %%
import matplotlib.pyplot as plt
import numpy as np

# Your plotting setup
fig, ax = plt.subplots(figsize=(5, 5))
FONT = 'Arial'
FONTSIZE = 21
plt.rc('font', family=FONT)
plt.rc('xtick', labelsize=FONTSIZE)
plt.rc('ytick', labelsize=FONTSIZE)
plt.rc('axes', labelsize=FONTSIZE)
plt.rc('legend', fontsize=FONTSIZE + 2)

# Define elements and color map
ELEMS = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br'}

from cycler import cycler
import matplotlib as mpl

# Get default color cycle from matplotlib
default_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']

# Assign default colors to elements
color_map = {ELEMS[key]: default_colors[i % len(default_colors)] for i, key in enumerate(ELEMS.keys())}

# Plot each element
for elem, symbol in ELEMS.items():
    idxs = (y == elem)
    ax.scatter(x_pca[idxs, 0], x_pca[idxs, 1], alpha=1, s=3, label=symbol, color=color_map[symbol])

# No ticks
ax.set_xticks([])
ax.set_yticks([])

# Labels and title
ax.set_xlabel(r'$u_1$', fontsize=FONTSIZE - 1)
ax.set_ylabel(r'$u_2$', fontsize=FONTSIZE - 1)
ax.set_title('Learned Atom Embeddings', fontsize=FONTSIZE)

# Function to overlay axes for independent legends
def overlay_axes_for_legend(fig, position):
    ax_leg = fig.add_axes(position)
    ax_leg.axis('off')
    return ax_leg

# Define groups for separate legends
element_groups = [
    ['H', 'C', 'N', 'O', 'F'],
    ['P', 'S', 'Cl'],
    ['Br']
]

# Position for the first legend group, adjust as necessary
legend_shift_y = -0.14
legend_shift_x = 0.165

legend_pos_x = 1.3
legend_pos_y = 0.7

colspace = 0.9

# Create and place each legend
for i, group in enumerate(element_groups):
    handles, labels = ax.get_legend_handles_labels()
    filtered_handles = [h for h, l in zip(handles, labels) if l in group]
    ax_leg = overlay_axes_for_legend(fig, [legend_pos_x, legend_pos_y, 0.2, 0.1])
    legend_pos_x += legend_shift_x
    legend_pos_y += legend_shift_y

    leg = ax_leg.legend(filtered_handles, group, frameon=False, handlelength=0.8, handletextpad=0., ncols=5,loc='upper center', bbox_to_anchor=(0.0, -0.0), columnspacing=colspace)

    for handle in leg.legendHandles:
        handle.set_sizes([80])  # Set to desired size
        handle.set_alpha(1)  # Set alpha

# Save the figure
name = 'pca_horizontal.png'
fig.savefig(name, dpi=300, bbox_inches='tight')

# %%
