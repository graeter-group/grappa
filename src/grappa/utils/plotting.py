#%%
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import copy
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple, Dict
from grappa.utils import unflatten_dict
import matplotlib.ticker as ticker


def scatter_plot(ax, x, y, n_max:int=None, seed=0, alpha:float=1., s:float=15, num_ticks:int=None, max_ticks:int=8, ax_symmetric=False, cluster=False, delta_factor=100, cmap='viridis', logscale=False, show_rmsd=False, amplitude=None, cbar_label:bool=False, **kwargs) -> plt.Axes:
    """
    Create a scatter plot of two arrays x and y.
    Args:
        ax: Matplotlib axis object
        x: Array of x values
        y: Array of y values
        n_max: Maximum number of points to plot
        seed: Random seed for selecting n_max points
        alpha: Transparency of the points
        s: Size of the points
        num_ticks: Number of ticks on the axes
        max_ticks: Maximum number of ticks on the axes
        ax_symmetric: Whether to make the axes symmetric
        cluster: Whether to cluster points
        delta_factor: Factor for clustering points
        cmap: Colormap for clustered scatter plot
        logscale: Whether to use a log scale for the colorbar
        show_rmsd: Whether to show the RMSD
        amplitude: Min and max val for symmetric axes
        **kwargs: Additional keyword arguments for ax.scatter call
    """
    if n_max is not None and n_max < len(x):
        np.random.seed(seed)
        idxs = np.random.choice(len(x), n_max, replace=False)
        x = x[idxs]
        y = y[idxs]

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    if ax_symmetric:
        if amplitude is not None:
            min_val = -amplitude
            max_val = amplitude
        else:   
            min_val = -max(abs(min_val), abs(max_val))
            max_val = max(abs(min_val), abs(max_val))

    ax.set_ylim(min_val, max_val)
    ax.set_xlim(min_val, max_val)

    ax.set_aspect('equal', 'box')

    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='-', linewidth=0.5)

    if num_ticks is not None:
        ax.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
        ax.yaxis.set_major_locator(plt.MaxNLocator(num_ticks))
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks))


    # Make the ticks go into and out of the plot and make them larger
    ax.tick_params(axis='both', which='major', direction='inout', length=10, width=1)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(-2,3))

    # Re-set the limits
    ax.set_ylim(min_val, max_val)
    ax.set_xlim(min_val, max_val)

    if cluster:
        points, frequencies = calculate_density_scatter(x, y, delta_factor=delta_factor, seed=seed)
        # revert the order of both to make the high frequency points appear on top:
        points = points[::-1]
        frequencies = frequencies[::-1]
        norm = plt.Normalize(vmin=min(frequencies), vmax=max(frequencies))

        if logscale:
            norm = colors.LogNorm(vmin=min(frequencies), vmax=max(frequencies))
        else:
            norm = plt.Normalize(vmin=min(frequencies), vmax=max(frequencies))
            
        sc = ax.scatter(points[:, 0], points[:, 1], c=frequencies, cmap=cmap, norm=norm, s=s, alpha=alpha, edgecolor='k', linewidths=0.5, **kwargs)

        # Create an axes divider for the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        # Add the colorbar
        cbar = plt.colorbar(sc, cax=cax)

        if max(frequencies) < 1000:
            # Use ScalarFormatter to avoid scientific notation and set ticks at integers only
            scalar_formatter = ticker.ScalarFormatter(useMathText=True)
            scalar_formatter.set_scientific(False)
            scalar_formatter.set_useOffset(False)
            cbar.ax.yaxis.set_major_formatter(scalar_formatter)

        if cbar_label:
            cbar.set_label("Frequency")

    else:
        ax.scatter(x, y, alpha=alpha, s=s, **kwargs)

    if show_rmsd:
        rmsd = np.sqrt(np.mean((x - y)**2))
        ax.text(0.05, 0.95, f'RMSD: {rmsd:.2f}', transform=ax.transAxes, ha='left', va='top')

    return ax


def calculate_density_scatter(x, y, delta_factor=100, seed=0):
    np.random.seed(seed)
    points = []
    frequencies = []

    # Create a deep copy of the input arrays
    x_copy = copy.deepcopy(x)
    y_copy = copy.deepcopy(y)

    # Calculate delta
    delta = max(max(x) - min(x), max(y) - min(y)) / delta_factor

    while len(x_copy) > 0:
        # Pick a random point
        idx = np.random.randint(len(x_copy))
        point_x, point_y = x_copy[idx], y_copy[idx]

        # Calculate distances to the random point
        distances = np.sqrt((x_copy - point_x)**2 + (y_copy - point_y)**2)

        # Find points within the distance delta
        within_delta = distances < delta

        # Count the number of points within delta
        frequency = np.sum(within_delta)

        # Store the point and its frequency
        points.append((point_x, point_y))
        frequencies.append(frequency)

        # Remove the points within delta from the copied arrays
        x_copy = x_copy[~within_delta]
        y_copy = y_copy[~within_delta]

    return np.array(points), np.array(frequencies)


def get_default_title_map():
    d = {
        'spice-dipeptide': 'SPICE Dipeptides',
        'spice-pubchem': 'SPICE PubChem',
        'spice-des-monomers': 'SPICE DES Monomers',
        'gen2': 'Gen2',
        'gen2-torsion': 'Gen2 Torsion',
        'rna-diverse': 'RNA Diverse',
        'rna-trinucleotide': 'RNA Trinucleotide',
        'pepconf-dlc': 'PepConf DLC',
        'protein-torsion': 'Protein Torsion',
        'dipeptides-300K-charmm36_nonb': 'Dipeptides 300K',
        'dipeptides-300K-charmm36': 'Dipeptides 300K',
        'dipeptides-300K-amber99': 'Dipeptides 300K',
        'dipeptides-300K-openff-1.2.0': 'Dipeptides 300K',
        'dipeptides-1000K-charmm36_nonb': 'Dipeptides 1000K',
        'dipeptides-1000K-charmm36': 'Dipeptides 1000K',
        'dipeptides-1000K-amber99': 'Dipeptides 1000K',
        'dipeptides-1000K-openff-1.2.0': 'Dipeptides 1000K',
        'uncapped-300K-amber99': 'Uncapped 300K',
        'uncapped-300K-openff-1.2.0': 'Uncapped 300K',
        'dipeptides-radical-300K': 'Radical Dipeptides 300K',
    }

    return d


def make_scatter_plots(ff_data, plot_dir=Path.cwd(), ylabel="Prediction", xlabel="QM", logscale:bool=True, ax_symmetric:Tuple[bool,bool]=(False,True), title_map:Dict[str,str]=get_default_title_map(), rmsd_position:Tuple[float,float]=(0.05, 0.95), cluster=True, dpi=200, figsize=5, **kwargs):
    """

    rmsd_position: Tuple[float,float]: Position of the RMSD text in the plot. If None, the RMSD is not shown.
    title_map: Dict[str,str]: Dictionary mapping dataset names to titles.
    """
    force_x_label = f"{xlabel} Force [kcal/mol/Å]"
    energy_x_label = f"{xlabel} Energy [kcal/mol]"

    plot_dir = Path(plot_dir)
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True, exist_ok=True)

    for dsname in ff_data['energies'].keys():
        ds_dir = plot_dir/dsname
        ds_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        scatter_plot(ax, ff_data['reference_energies'][dsname], ff_data['energies'][dsname], logscale=logscale, ax_symmetric=ax_symmetric[0], cluster=cluster, **kwargs)
        ax.set_xlabel(energy_x_label)
        ax.set_ylabel(ylabel)

        if dsname in title_map:
            ax.set_title(title_map[dsname])

        if rmsd_position is not None:
            rmsd_value = ((ff_data['reference_energies'][dsname] - ff_data['energies'][dsname])**2).mean()**0.5
            ax.text(rmsd_position[0], rmsd_position[1], f'RMSD: {rmsd_value:.2f}', transform=ax.transAxes, verticalalignment='top')
    
        plt.tight_layout()
        plt.savefig(ds_dir/"energy.png", dpi=dpi)
        plt.close()
    

    for dsname in ff_data['gradients'].keys():
        ds_dir = plot_dir/dsname
        ds_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        scatter_plot(ax, ff_data['reference_gradients'][dsname].flatten(), ff_data['gradients'][dsname].flatten(), logscale=logscale, ax_symmetric=ax_symmetric[1], cluster=cluster, **kwargs)
        ax.set_xlabel(force_x_label)
        ax.set_ylabel(ylabel)

        if dsname in title_map:
            ax.set_title(title_map[dsname])

        if rmsd_position is not None:
            rmsd_value = ((ff_data['reference_gradients'][dsname].flatten() - ff_data['gradients'][dsname].flatten())**2).mean()**0.5
            ax.text(rmsd_position[0], rmsd_position[1], f'RMSD: {rmsd_value:.2f}', transform=ax.transAxes, verticalalignment='top')
    
        plt.tight_layout()
        plt.savefig(ds_dir/"force.png", dpi=dpi)
        plt.close()


    for dsname in ff_data['energies'].keys():
        ds_dir = plot_dir/dsname
        ds_dir.mkdir(parents=True, exist_ok=True)
        all_energies = ff_data['energies'][dsname]
        all_ref_energies = ff_data['reference_energies'][dsname]
        mol_idxs = ff_data['mol_idxs'][dsname]
            
        energies_per_mol = [all_energies[mol_idxs[i]:mol_idxs[i+1]] for i in range(len(mol_idxs)-1)] + [all_energies[mol_idxs[-1]:]]
        ref_energies_per_mol = [all_ref_energies[mol_idxs[i]:mol_idxs[i+1]] for i in range(len(mol_idxs)-1)] + [all_ref_energies[mol_idxs[-1]:]]

        rmsd_values = [((np.array(ref_energies_per_mol[i]) - np.array(energies_per_mol[i]))**2).mean()**0.5 for i in range(len(energies_per_mol))]

        # Histogram of the energy errors
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        ax.hist(rmsd_values, bins=30)
        ax.set_xlabel("RMSD [kcal/mol]")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Energy RMSD per molecule" + (f" ({title_map[dsname]})" if dsname in title_map else ""))

        plt.tight_layout()
        plt.savefig(ds_dir/"energy_rmsd_histogram.png", dpi=dpi)
        plt.close()

    for dsname in ff_data['gradients'].keys():
        ds_dir = plot_dir/dsname
        ds_dir.mkdir(parents=True, exist_ok=True)
        all_gradients = ff_data['gradients'][dsname]
        all_ref_gradients = ff_data['reference_gradients'][dsname]
        mol_idxs = ff_data['mol_idxs'][dsname]

        gradients_per_mol = [all_gradients[mol_idxs[i]:mol_idxs[i+1]] for i in range(len(mol_idxs)-1)] + [all_gradients[mol_idxs[-1]:]]
        ref_gradients_per_mol = [all_ref_gradients[mol_idxs[i]:mol_idxs[i+1]] for i in range(len(mol_idxs)-1)] + [all_ref_gradients[mol_idxs[-1]:]]

        rmsd_values = [((np.array(ref_gradients_per_mol[i]).flatten() - np.array(gradients_per_mol[i]).flatten())**2).mean()**0.5 for i in range(len(gradients_per_mol))]

        # Histogram of the force errors
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        ax.hist(rmsd_values, bins=30)
        ax.set_xlabel("RMSD [kcal/mol/Å]")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Force RMSD per molecule" + (f" ({title_map[dsname]})" if dsname in title_map else ""))

        plt.tight_layout()
        plt.savefig(ds_dir/"force_rmsd_histogram.png", dpi=dpi)
        plt.close()


#%%

# example:
if __name__ == '__main__':

    # generate points along the diagonal with noise:
    x = np.linspace(0, 10, 100000)
    y = x + np.random.randn(100000)

    fig, ax = plt.subplots()
    scatter_plot(ax, x, y, cluster=True, delta_factor=100)
    plt.show()
    fig, ax = plt.subplots()
    scatter_plot(ax, x, y, cluster=False, s=1)
    plt.show()

    #%%
    # test:
    dspath = "/local/user/seutelf/grappa/ckpt/grappa-1.3/baseline/2024-06-19_04-54-30/test_data/epoch-784.npz"
    data = np.load(dspath)
    data = unflatten_dict(data)
    data.keys()
    #%%
    dsname = 'dipeptides-300K-charmm36'
    all_energies = data['energies'][dsname]
    all_ref_energies = data['reference_energies'][dsname]
    mol_idxs = data['mol_idxs'][dsname]

    mol_idxs[:5]
    #%%

    energies_per_mol = [all_energies[mol_idxs[i]:mol_idxs[i+1]] for i in range(len(mol_idxs)-1)] + [all_energies[mol_idxs[-1]:]]
    ref_energies_per_mol = [all_ref_energies[mol_idxs[i]:mol_idxs[i+1]] for i in range(len(mol_idxs)-1)] + [all_ref_energies[mol_idxs[-1]:]]
    rmsd_values = [((np.array(ref_energies_per_mol[i]) - np.array(energies_per_mol[i]))**2).mean()**0.5 for i in range(len(energies_per_mol))]
    len(rmsd_values)
    ref_energies_per_mol[2]
    #%%
    make_scatter_plots(data, plot_dir=Path(__file__).parent/'plots')

# %%
