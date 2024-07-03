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
import logging


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

    if num_ticks is not None:
        ax.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
        ax.yaxis.set_major_locator(plt.MaxNLocator(num_ticks))
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks))
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=max_ticks))


    # Make the ticks go into and out of the plot and make them larger
    ax.tick_params(axis='both', which='major', direction='inout', length=10, width=1)
    ax.ticklabel_format(style='sci', axis='both', scilimits=(-2,3))

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

    # limits and ticks:
    ax.set_aspect('equal', 'box')

    x_min, y_min = ax.get_xlim()[0], ax.get_ylim()[0]
    x_max, y_max = ax.get_xlim()[1], ax.get_ylim()[1]

    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    if ax_symmetric:
        if amplitude is not None:
            min_val = -amplitude
            max_val = amplitude
        else:
            min_val = -max(abs(min_val), abs(max_val))
            max_val = max(abs(min_val), abs(max_val))

    # reference line
    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='-', linewidth=0.5)

    ax.set_ylim(min_val, max_val)
    ax.set_xlim(min_val, max_val)


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


def make_scatter_plots(ff_data, plot_dir=Path.cwd(), ylabel="Prediction", xlabel="QM", logscale:bool=True, ax_symmetric:Tuple[bool,bool]=(False,True), title_map:Dict[str,str]=get_default_title_map(), rmsd_position:Tuple[float,float]=(0.05, 0.95), cluster=True, dpi=200, figsize=5, contributions=['bond', 'angle', 'proper', 'improper', 'nonbonded', 'total'], **kwargs):
    """

    rmsd_position: Tuple[float,float]: Position of the RMSD text in the plot. If None, the RMSD is not shown.
    title_map: Dict[str,str]: Dictionary mapping dataset names to titles.
    """

    plot_dir = Path(plot_dir)
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True, exist_ok=True)

    DSNAMES = list(set(list(ff_data['energies'].keys()) + list(ff_data['gradients'].keys())))


    for dsname in tqdm(DSNAMES, desc='Creating plots for datasets'):

        # SCATTER PLOTS ENERGY
        skip = False
        if dsname not in ff_data['energies']:
            skip = True
        if not skip:
            ds_dir = plot_dir/dsname
            ds_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(figsize, figsize))
            scatter_plot(ax, ff_data['reference_energies'][dsname], ff_data['energies'][dsname], logscale=logscale, ax_symmetric=ax_symmetric[0], cluster=cluster, **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            if dsname in title_map:
                ax.set_title(title_map[dsname] + " - Energy [kcal/mol]")
            else:
                ax.set_title(dsname + " - Energy [kcal/mol]")

            if rmsd_position is not None:
                rmsd_value = ((ff_data['reference_energies'][dsname] - ff_data['energies'][dsname])**2).mean()**0.5
                ax.text(rmsd_position[0], rmsd_position[1], f'RMSD: {rmsd_value:.2f}', transform=ax.transAxes, verticalalignment='top')
        
            plt.tight_layout()
            plt.savefig(ds_dir/"energy.png", dpi=dpi)
            plt.close()
    

        # SCATTER PLOTS FORCE
        skip = False
        if dsname not in ff_data['gradients']:
            skip = True
        if not skip:
            ds_dir = plot_dir/dsname
            ds_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(figsize, figsize))
            scatter_plot(ax, -ff_data['reference_gradients'][dsname].flatten(), -ff_data['gradients'][dsname].flatten(), logscale=logscale, ax_symmetric=ax_symmetric[1], cluster=cluster, **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            if dsname in title_map:
                ax.set_title(title_map[dsname] + " - Force [kcal/mol/Å]")
            else:
                ax.set_title(dsname + " - Force [kcal/mol/Å]")

            if rmsd_position is not None:
                rmsd_value = ((ff_data['reference_gradients'][dsname].flatten() - ff_data['gradients'][dsname].flatten())**2).mean()**0.5
                ax.text(rmsd_position[0], rmsd_position[1], f'RMSD: {rmsd_value:.2f}', transform=ax.transAxes, verticalalignment='top')
        
            plt.tight_layout()
            plt.savefig(ds_dir/"force.png", dpi=dpi)
            plt.close()


        # HISTOGRAMS ENERGY
        skip = False
        if not dsname in ff_data['energies']:
            skip = True

        if not skip:
            ds_dir = plot_dir/dsname
            ds_dir.mkdir(parents=True, exist_ok=True)
            all_energies = ff_data['energies'][dsname]
            all_ref_energies = ff_data['reference_energies'][dsname]
            energy_mol_idxs = ff_data['energy_mol_idxs'][dsname]
                
            energies_per_mol = [all_energies[energy_mol_idxs[i]:energy_mol_idxs[i+1]] for i in range(len(energy_mol_idxs)-1)]
            ref_energies_per_mol = [all_ref_energies[energy_mol_idxs[i]:energy_mol_idxs[i+1]] for i in range(len(energy_mol_idxs)-1)]

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


        # HISTOGRAMS FORCE
        skip = False
        if not dsname in ff_data['gradients']:
            skip = True

        if not skip:
            ds_dir = plot_dir/dsname
            ds_dir.mkdir(parents=True, exist_ok=True)
            all_gradients = ff_data['gradients'][dsname]
            all_ref_gradients = ff_data['reference_gradients'][dsname]
            gradient_mol_idxs = ff_data['gradient_mol_idxs'][dsname]

            gradients_per_mol = [all_gradients[gradient_mol_idxs[i]:gradient_mol_idxs[i+1]] for i in range(len(gradient_mol_idxs)-1)]
            ref_gradients_per_mol = [all_ref_gradients[gradient_mol_idxs[i]:gradient_mol_idxs[i+1]] for i in range(len(gradient_mol_idxs)-1)]

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

        def remove_percentile(data, percentile):
            threshold = np.percentile(data, percentile)
            return data[data <= threshold]

        # ABS CONTRIBUTION VIOLIN PLOTS
        skip = False
        if 'gradient_contributions' not in ff_data or dsname not in ff_data['gradient_contributions']:
            skip = True
        if not skip:
            contrib_dict = ff_data['gradient_contributions'][dsname]
            if len(list(contrib_dict.keys())) == 0:
                skip = True
            if not skip:
                ds_dir = plot_dir / dsname
                ds_dir.mkdir(parents=True, exist_ok=True)
                contribs_present = [c for c in contributions if c in contrib_dict.keys()]

                num_contribs = len(contribs_present)

                fig, ax = plt.subplots(figsize=(figsize/4*num_contribs, figsize))

                all_norms = []
                labels = []
                for contrib_name in contribs_present:
                    contrib = contrib_dict[contrib_name]
                    norm = np.sqrt((contrib**2).sum(axis=-1))
                    norm = remove_percentile(norm, 99)
                    all_norms.append(norm)
                    labels.append(contrib_name.capitalize())

                # Create the violin plot
                parts = ax.violinplot(all_norms, showmeans=True, showextrema=True, points=1000, bw_method=0.05)

                # Set x-axis labels
                ax.set_xticks(np.arange(1, len(labels) + 1))
                ax.set_xticklabels(labels)
                ax.set_ylabel("Force L2 Norm [kcal/mol/Å]")
                ax.set_title(f"Atomic Forces per Contribution")

                # add some padding:
                plt.savefig(ds_dir/"contributions.png", dpi=dpi)
                plt.close()



def compare_scatter_plots(ff_data_1, ff_data_2, plot_dir=Path.cwd(), ylabel="Prediction 2", xlabel="Prediction 1", logscale:bool=True, ax_symmetric:Tuple[bool,bool]=(False,True), title_map:Dict[str,str]=get_default_title_map(), rmsd_position:Tuple[float,float]=(0.05, 0.95), cluster=True, dpi=200, figsize=5, contributions=['bond', 'angle', 'proper', 'improper', 'nonbonded', 'total'], **kwargs):
    """
    Compare scatter plots of two datasets.

    rmsd_position: Tuple[float,float]: Position of the RMSD text in the plot. If None, the RMSD is not shown.
    title_map: Dict[str,str]: Dictionary mapping dataset names to titles.
    """
    plot_dir = Path(plot_dir)
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True, exist_ok=True)

    DSNAMES = list(set(list(ff_data_1['energies'].keys()) + list(ff_data_1['gradients'].keys())))

    for dsname in tqdm(DSNAMES, desc='Creating comparison plots'):
        # SCATTER PLOTS ENERGY
        skip = False
        if dsname not in ff_data_1['energies'] or dsname not in ff_data_2['energies']:
            skip = True

        if not skip:
            ds_dir = plot_dir / dsname
            ds_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(figsize, figsize))
            scatter_plot(ax, ff_data_1['energies'][dsname], ff_data_2['energies'][dsname], logscale=logscale, ax_symmetric=ax_symmetric[0], cluster=cluster, **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            if dsname in title_map:
                ax.set_title(title_map[dsname] + " - Energy [kcal/mol]")
            else:
                ax.set_title(dsname + " - Energy [kcal/mol]")

            if rmsd_position is not None:
                rmsd_value = ((ff_data_1['energies'][dsname] - ff_data_2['energies'][dsname])**2).mean()**0.5
                ax.text(rmsd_position[0], rmsd_position[1], f'RMSD: {rmsd_value:.2f}', transform=ax.transAxes, verticalalignment='top')

            plt.tight_layout()
            plt.savefig(ds_dir / "energy_comparison.png", dpi=dpi)
            plt.close()

    
        # SCATTER PLOTS FORCE
        skip = False
        if dsname not in ff_data_1['gradients'] or dsname not in ff_data_2['gradients']:
            skip = True

        if not skip:
            ds_dir = plot_dir / dsname
            ds_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(figsize, figsize))
            scatter_plot(ax, -ff_data_1['gradients'][dsname].flatten(), -ff_data_2['gradients'][dsname].flatten(), logscale=logscale, ax_symmetric=ax_symmetric[1], cluster=cluster, **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            if dsname in title_map:
                ax.set_title(title_map[dsname] + " - Force [kcal/mol/Å]")
            else:
                ax.set_title(dsname + " - Force [kcal/mol/Å]")

            if rmsd_position is not None:
                rmsd_value = ((ff_data_1['gradients'][dsname].flatten() - ff_data_2['gradients'][dsname].flatten())**2).mean()**0.5
                ax.text(rmsd_position[0], rmsd_position[1], f'RMSD: {rmsd_value:.2f}', transform=ax.transAxes, verticalalignment='top')

            plt.tight_layout()
            plt.savefig(ds_dir / "force_comparison.png", dpi=dpi)
            plt.close()

        # CONTRIBUTION COMPARISON
        skip = False
        if dsname not in ff_data_1['gradient_contributions'] or dsname not in ff_data_2['gradient_contributions']:
            skip = True
        
        if not skip:
            contrib_dict_1 = ff_data_1['gradient_contributions'][dsname]

            contrib_dict_2 = ff_data_2['gradient_contributions'][dsname]

            contribs_present = [c for c in contributions if c in contrib_dict_1.keys() and c in contrib_dict_2.keys()]

            num_contribs = len(contribs_present)

            if num_contribs <= 1:
                skip = True

            if not skip:
                ds_dir = plot_dir / dsname
                ds_dir.mkdir(parents=True, exist_ok=True)

                num_cols = min(num_contribs, 3)
                num_rows = num_contribs // num_cols + (1 if num_contribs % num_cols > 0 else 0)

                fig, axs = plt.subplots(num_rows, num_cols, figsize=(figsize*num_cols, figsize*num_rows))

                if isinstance(axs, np.ndarray):
                    axs = axs.flatten()

                for i, contrib_name in enumerate(contribs_present):
                    ax = axs[i]
                    contrib_1 = contrib_dict_1[contrib_name].flatten()
                    contrib_2 = contrib_dict_2[contrib_name].flatten()
                    if not len(contrib_1) == len(contrib_2):
                        logging.warning(f"Length of contributions {contrib_name} in {dsname} does not match. Skipping.")
                        continue

                    ax = scatter_plot(ax, contrib_1, contrib_2, logscale=logscale, ax_symmetric=ax_symmetric[1], cluster=cluster, **kwargs)

                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    ax.set_title(contrib_name.capitalize())

                    if rmsd_position is not None:
                        rmsd_value = ((contrib_1 - contrib_2)**2).mean()**0.5
                        ax.text(rmsd_position[0], rmsd_position[1], f'RMSD: {rmsd_value:.2f}', transform=ax.transAxes, verticalalignment='top')

                dsname = title_map[dsname] if dsname in title_map else dsname
                plt.suptitle(f'{dsname} - Force Contributions [kcal/mol/Å]')

                plt.tight_layout(pad=2)

                plt.savefig(ds_dir / "contribution_comparison.png", dpi=dpi)
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
    dspath = "/local/user/seutelf/grappa/ckpt/grappa-1.3/published/2024-06-26_01-30-36/test_data/epoch:789.npz"
    data = np.load(dspath)
    data = unflatten_dict(data)
    data.keys()
    #%%
    dsname = 'uncapped-300K-amber99'
    all_energies = data['energies'][dsname]
    all_ref_energies = data['reference_energies'][dsname]
    energy_mol_idxs = data['energy_mol_idxs'][dsname]
    gradient_mol_idxs = data['gradient_mol_idxs'][dsname]

    print(data['gradient_contributions'][dsname]['bond'].shape)
    print(data['gradients'][dsname].shape)

    #%%

    energies_per_mol = [all_energies[energy_mol_idxs[i]:energy_mol_idxs[i+1]] for i in range(len(energy_mol_idxs)-1)]
    ref_energies_per_mol = [all_ref_energies[energy_mol_idxs[i]:energy_mol_idxs[i+1]] for i in range(len(energy_mol_idxs)-1)]
    rmsd_values = [((np.array(ref_energies_per_mol[i]) - np.array(energies_per_mol[i]))**2).mean()**0.5 for i in range(len(energies_per_mol))]
    len(rmsd_values)
    print([e.shape for e in energies_per_mol])
    #%%
    make_scatter_plots(data, plot_dir=Path(__file__).parent/'plots')
    #%%
    dspath2 = "/local/user/seutelf/grappa/ckpt/grappa-1.3/published/2024-06-26_01-30-36/test_data/amber99sbildn/data.npz"
    # dspath2=dspath
    data_amber = np.load(dspath2)
    data_amber = unflatten_dict(data_amber)
    print(data_amber['gradient_contributions'][dsname].keys())
    compare_scatter_plots(data, data_amber, plot_dir=Path(__file__).parent/'compare_plots_amber', xlabel='Grappa', ylabel='FF99SB-ILDN')
# %%
