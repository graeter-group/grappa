#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import LogFormatter, LogLocator, ScalarFormatter
import matplotlib
import json

#%%
data = json.load(open('new_results.json', 'r'))
# data_classical = json.load(open('../grappa-1.1/results.json', 'r'))
data_espaloma = json.load(open('../espaloma_benchmark/espaloma_test_results.json', 'r'))

data_conventional = json.load(open('../espaloma_benchmark/results.json', 'r'))

def make_lc_plot(data, data_espaloma, data_conventional, dataset, ax=None, conventional_ff='gaff-2.11', ylabel=True, legend=True, ff_name='Gaff 2.11', title=None, only_conv_in_legend=False, ylim=None):
    if ax is None:
        fig, ax = plt.subplots(2,1, figsize=(6, 6))
    
    FONTSIZE = 18
    FONT = 'Arial'

    GRAPPA_COLOR = '#1f77b4'
    AMBER_COLOR = '#e41a1c'

    LOG = False

    plt.rc('font', family=FONT)
    plt.rc('xtick', labelsize=FONTSIZE)
    plt.rc('ytick', labelsize=FONTSIZE)
    plt.rc('axes', labelsize=FONTSIZE)
    plt.rc('legend', fontsize=FONTSIZE)

    num_mols = [data[k]['train_mols'] for k in sorted(data.keys())]
    grappa_energy_rmse = [data[k][dataset]['rmse_energies']['mean'] for k in sorted(data.keys())]
    grappa_gradient_crmse = [data[k][dataset]['crmse_gradients']['mean'] for k in sorted(data.keys())]

    # create np arrays, sort by num_mols and create 2 plots, one with energy rmse and one with gradient crmse against num_mols
    num_mols, grappa_energy_rmse, grappa_gradient_crmse = zip(*sorted(zip(num_mols, grappa_energy_rmse, grappa_gradient_crmse)))

    num_mols = np.array(num_mols)
    grappa_energy_rmse = np.array(grappa_energy_rmse)
    grappa_gradient_crmse = np.array(grappa_gradient_crmse)

    espaloma_energy_rmse = data_espaloma[dataset]['rmse_energies']['mean']
    espaloma_gradient_crmse = data_espaloma[dataset]['crmse_gradients']['mean']

    conventional_energy_rmse = data_conventional['test'][dataset][conventional_ff]['rmse_energies']['mean']
    conventional_gradient_crmse = data_conventional['test'][dataset][conventional_ff]['crmse_gradients']['mean']

    ax[0].axhline(conventional_energy_rmse, label=ff_name, color=AMBER_COLOR, linestyle='--')
    ax[1].axhline(conventional_gradient_crmse, label=ff_name, color=AMBER_COLOR, linestyle='--')

    ax[0].scatter(num_mols, grappa_energy_rmse, label='Grappa', color=GRAPPA_COLOR)
    ax[1].scatter(num_mols, grappa_gradient_crmse, label='Grappa', color=GRAPPA_COLOR)

    ax[0].plot(num_mols, grappa_energy_rmse, color=GRAPPA_COLOR)
    ax[1].plot(num_mols, grappa_gradient_crmse, color=GRAPPA_COLOR)

    ax[0].axhline(espaloma_energy_rmse, label='Espaloma', color='green', linestyle='--')
    ax[1].axhline(espaloma_gradient_crmse, label='Espaloma', color='green', linestyle='--')

    ylim0, _ = ax[0].get_ylim()
    if ylim is not None:
        ax[0].set_ylim(ylim0, ylim)

    ax[1].set_xlabel('Training molecules')
    if ylabel:
        ax[0].set_ylabel('Energy RMSE\n[kcal/mol]')
        ax[1].set_ylabel('Force cRMSE\n[kcal/mol/Å]')

    if legend:
        if only_conv_in_legend:
            ax[0].legend([f'{ff_name}'], frameon=True)
        else:
            ax[0].legend(frameon=True)

    if title is not None:
        ax[0].set_title(title, fontsize=FONTSIZE+2)

    # turn off x ticks labelling for the first plot:
    ax[0].set_xticklabels([])

    # make a grid:
    ax[0].grid(alpha=0.4)
    ax[1].grid(alpha=0.4)

    # ax[1].set_xscale('log')
    # ax[0].set_xscale('log')
    
    # set logscales:
    if LOG:
        formatter = ScalarFormatter()

        ax[0].yaxis.set_major_formatter(formatter)

        ax[1].yaxis.set_major_formatter(formatter)
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')

    return ax

#%%
make_lc_plot(data, data_espaloma, data_conventional, 'spice-dipeptide', conventional_ff='amber14', ff_name='FF14SB')
# %%
make_lc_plot(data, data_espaloma, data_conventional, 'spice-pubchem', only_conv_in_legend=True)
# %%
make_lc_plot(data, data_espaloma, data_conventional, 'rna-trinucleotide', conventional_ff='amber14', ff_name='RNA.OL3')

# %%
fig, ax = plt.subplots(2, 3, figsize=(12, 5))

make_lc_plot(data, data_espaloma, data_conventional, 'spice-pubchem', ax=ax[:,0], ylabel=True, legend=True, title='Small molecules', ff_name='Gaff 2.11', conventional_ff='gaff-2.11')
make_lc_plot(data, data_espaloma, data_conventional, 'spice-dipeptide', ax=ax[:,1], ylabel=False, legend=True, title='Dipeptides', ff_name='FF14SB', conventional_ff='amber14', only_conv_in_legend=True)
make_lc_plot(data, data_espaloma, data_conventional, 'rna-trinucleotide', ax=ax[:,2], ylabel=False, legend=True, title='Trinucleotide', ff_name='RNA.OL3', conventional_ff='amber14', only_conv_in_legend=True, ylim=7.5)

fig.tight_layout()
fig.savefig('lc_benchmark.png', dpi=400, bbox_inches='tight')
# %%
