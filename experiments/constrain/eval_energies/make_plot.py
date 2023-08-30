#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json

# %%

data_large_peptides = json.load(open("energy_force_rmse.json"))
data_qm_tripeptides = json.load(open("tripep_QM_energy_force_rmse.json"))
# %%
param_weight = np.array([1,10,100,1000])
forces_qm_rmse = np.array([data_qm_tripeptides[str(p)][1] for p in param_weight])
forces_rmse = np.array([data_large_peptides[str(p)][1] for p in param_weight])
# %%
# make a plot with log x axis whith p weight on the x axis and force rmse on the y axis.
# Setup for nice fonts, title, and labels
mpl.rc('font', family='Arial', size=16)
plt.figure(figsize=(10, 6))

# Plot the data
plt.plot(param_weight, forces_rmse, marker='o', linestyle='--', label='Large Peptides: Diff')
plt.plot(param_weight, forces_qm_rmse, marker='o', linestyle='--', label='Tripeptides: Grappa')

with open("amber_tripep_QM_energy_force_rmse.json", "r") as f:
    amber_error = json.load(f)[1]

plt.hlines(amber_error, 0, 1000, linestyle='--', label='Tripeptides: Amber ff', colors="red", alpha=0.5)

# Customize axes to be logarithmic, set labels, and title
plt.xscale("log")
plt.xlabel("Parameter Weight")
plt.ylabel("Force RMSE [kcal/mol/Å]")
plt.title("Force RMSE on Tripeptides (QM) and Large Peptides")

# Show legend
# shift it slightly down
plt.legend(loc="upper right", bbox_to_anchor=(1, 0.95))

# Save or show the plot
plt.savefig("force_rmse_vs_param_weight.png", dpi=500)
plt.show()
# %%
# now the same for energies...:
# Load the energy data
energies_qm_rmse = np.array([data_qm_tripeptides[str(p)][0] for p in param_weight])
energies_rmse = np.array([data_large_peptides[str(p)][0] for p in param_weight])

# Setup the plot
mpl.rc('font', family='Arial', size=16)
plt.figure(figsize=(10, 6))

# Plot the data
plt.plot(param_weight, energies_rmse, marker='o', linestyle='--', label='Large Peptides: Diff')
plt.plot(param_weight, energies_qm_rmse, marker='o', linestyle='--', label='Tripeptides: Grappa')

# If you have Amber ff data for energies, you can include it like before
with open("amber_tripep_QM_energy_force_rmse.json", "r") as f:
    amber_energy_error = json.load(f)[0]

plt.hlines(amber_energy_error, 0, 1000, linestyle='--', label='Tripeptides: Amber ff', colors="red", alpha=0.5)

# Customize axes, labels, and title
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Parameter Weight")
plt.ylabel("Energy RMSE [kcal/mol]")
plt.title("Energy RMSE on Tripeptides (QM) and Large Peptides")

# Show legend and adjust its position
plt.legend(loc="best")

# Save or show the plot
plt.savefig("energy_rmse_vs_param_weight.png", dpi=500)
plt.show()
# %%
mpl.rc('font', family='Arial', size=16)
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot forces
axs[0].plot(param_weight, forces_rmse, marker='o', linestyle='--', label='Large Peptides: Diff')
axs[0].plot(param_weight, forces_qm_rmse, marker='o', linestyle='--', label='Tripeptides: Grappa')

with open("amber_tripep_QM_energy_force_rmse.json", "r") as f:
    amber_error = json.load(f)[1]

axs[0].hlines(amber_error, 0, 1000, linestyle='--', label='Tripeptides: Amber ff', colors="red", alpha=0.5)

# Customize the first subplot
axs[0].set_xscale("log")
axs[0].set_xlabel("Parameter Weight")
axs[0].set_ylabel("Force RMSE [kcal/mol/Å]")
axs[0].set_title("Force RMSE")
axs[0].legend(loc="best")

# Plot energies
axs[1].plot(param_weight, energies_rmse, marker='o', linestyle='--', label='Large Peptides: Diff')
axs[1].plot(param_weight, energies_qm_rmse, marker='o', linestyle='--', label='Tripeptides: Grappa')

with open("amber_tripep_QM_energy_force_rmse.json", "r") as f:
    amber_energy_error = json.load(f)[0]

axs[1].hlines(amber_energy_error, 0, 1000, linestyle='--', label='Tripeptides: Amber ff', colors="red", alpha=0.5)

# Customize the second subplot
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_xlabel("Parameter Weight")
axs[1].set_ylabel("Energy RMSE [kcal/mol]")
axs[1].set_title("Energy RMSE")
axs[1].legend(loc="best")

# Finalize and save or show the plot
plt.tight_layout()
plt.savefig("rmse_vs_param_weight_subplots.png", dpi=500)
plt.show()
# %%
