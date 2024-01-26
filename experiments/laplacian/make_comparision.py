#%%
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
PROJECT = "esp_split_hybrid"
DATASETS = ["gen2", "gen2-torsion", "pepconf-dlc", "protein-torsion", "spice-pubchem", "spice-dipeptide", "spice-des-monomers", "rna-diverse"]
K = 3  # Desired number of best runs to select

FILTER_SAME_NAME = True  # Filter out runs with the same name

#%%
# PARSE ESPALOMA REPORT SUMMARY
with open("report_summary.csv", "r") as f:
    data = f.read()

# Parse the data
parsed_data = {}
lines = data.split('\n')
for i in range(0, len(lines)):
    if "(vl)" in lines[i]:
        dataset = lines[i].split()[0][1:]
        energy = float(lines[i+2].split()[1].split('_')[0][1:])
        force = float(lines[i+3].split()[1].split('_')[0][1:])
        parsed_data[dataset] = {'energy': energy, 'force': force}
#%%

REFERENCE_VALUES = parsed_data

# Initialize W&B API
api = wandb.Api()

# Fetch runs
runs = api.runs(PROJECT)

# Collect metrics
metrics_data = {}

for run in runs:
    name = run.name + "-" + run.id
    given_name = run.name.split('-')[3:]
    given_name = '-'.join(given_name)
    print(given_name)
    history = run.history(samples=1000)
    if not history.empty:
        early_stopping_losses = history['early_stopping_loss'].values
        # find the index of the best validation loss:
        best_val_loss = np.nanmin(early_stopping_losses)
        best_val_loss_idx = np.nanargmin(early_stopping_losses)

        metrics_data[name] = {'best_val_loss': best_val_loss, 'given_name': given_name}

        for ds in DATASETS:
            force_crmse = history[f'{ds}/val/crmse_gradients'].values[best_val_loss_idx]
            energy_rmse = history[f'{ds}/val/rmse_energies'].values[best_val_loss_idx]
            metrics_data[name] = metrics_data.get(name, {})
            metrics_data[name][ds] = {
                'force_crmse': force_crmse,
                'energy_rmse': energy_rmse,
            }
metrics_data
#%%

# font = 'Arial'
# plt.rc('font', family=font)
fontsize = 16
plt.rc('font', size=fontsize)
            
# now do a bar plot of the best k runs and the reference with the ds as x axis
# the run names are encoded as color
            
# Sort runs based on best validation loss and select top K
sorted_runs = sorted(metrics_data.items(), key=lambda x: x[1]['best_val_loss'])

if FILTER_SAME_NAME:
    import copy
    sorted_runs_copy = copy.deepcopy(sorted_runs)
    sorted_runs = []
    # Only keep the best run in sorted runs for each given name, also replace the name by the given name
    given_names = set()
    for run_name, metrics in sorted_runs_copy:
        given_name = metrics['given_name']
        if given_name in given_names:
            continue
        given_names.add(given_name)
        sorted_runs.append((given_name, metrics))
        
#%%

if K <= len(sorted_runs):
    top_k_runs = sorted_runs[:K]
else:
    top_k_runs = sorted_runs

# Prepare data for plotting
ds_metrics = {ds: {'force_crmse': [REFERENCE_VALUES[ds]['force']], 'energy_rmse': [REFERENCE_VALUES[ds]['energy']]} for ds in DATASETS}
# choose K+1 run colors from matplotlibs standard color cycle
run_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:K+1]
run_names = ['Espaloma']

for run_name, metrics in top_k_runs:
    run_names.append(run_name)
    for ds in DATASETS:
        ds_metrics[ds]['force_crmse'].append(metrics[ds]['force_crmse'])
        ds_metrics[ds]['energy_rmse'].append(metrics[ds]['energy_rmse'])

#%%

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

num_datasets = len(DATASETS)
bar_width = 0.8  # Width of the bars in the plot

num_bars = len(top_k_runs) + 1

# Plotting Energy RMSE and Force cRMSE for each dataset and run
x_ticks = []
for dataset_index, dataset_name in enumerate(DATASETS):
    # Energy RMSE for current dataset across top K runs
    energy_rmse_values = ds_metrics[dataset_name]['energy_rmse']

    offset = dataset_index * ((num_bars) / num_datasets + 1 + num_bars) # Offset for each dataset's bars

    x_pos = np.arange(num_bars) + offset

    x_ticks.append(x_pos.mean())

    # Plotting Energy RMSE for the current dataset across top K runs
    axs[0].bar(x_pos, np.array(energy_rmse_values), width=bar_width, color=run_colors)

    # Force cRMSE for current dataset across top K runs
    force_crmse_values = ds_metrics[dataset_name]['force_crmse']
    axs[1].bar(x_pos, force_crmse_values, width=bar_width, color=run_colors)

# Setting Labels, Titles, and Legends
axs[0].set_ylabel('Energy RMSE')
axs[1].set_ylabel('Force cRMSE')
axs[0].set_title('Energy RMSE Comparison')
axs[1].set_title('Force cRMSE Comparison')

# Adjusting x-axis for Run Names
# plt.xlabel('Dataset')
axs[1].set_xticks(x_ticks)
axs[1].set_xticklabels(DATASETS, rotation=30, fontsize=fontsize-3)


# Adding Legend for the color manually:
handles = []
for i in range(len(run_names)):
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=run_colors[i]))
axs[0].legend(handles, [name[:10] for name in run_names], loc='upper left')

# Adjusting Layout and Displaying Plot
plt.tight_layout()
plt.savefig('comparison.png')
# %%
