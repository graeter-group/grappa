#%%
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
import copy
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def scatter_plot(ax, x, y, n_max=None, seed=0, symmetric=False, alpha=1., s=15, num_ticks=None, ax_symmetric=False, cluster=False, delta_factor=100, cmap='viridis', logscale=False, **kwargs):
    """
    Create a scatter plot of two arrays x and y.
    Args:
        ax: Matplotlib axis object
        x: Array of x values
        y: Array of y values
        n_max: Maximum number of points to plot
        seed: Random seed for selecting n_max points
        symmetric: Make the plot symmetric around the origin
        alpha: Transparency of the points
        s: Size of the points
        num_ticks: Number of ticks on the axes
        ax_symmetric: Whether to make the axes symmetric
        cluster: Whether to cluster points
        delta_factor: Factor for clustering points
        cmap: Colormap for clustered scatter plot
        logscale: Whether to use a log scale for the colorbar
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
        min_val = -max(abs(min_val), abs(max_val))
        max_val = max(abs(min_val), abs(max_val))

    if symmetric:
        val = max(abs(min_val), abs(max_val))
        min_val = -val
        max_val = val

    ax.set_ylim(min_val, max_val)
    ax.set_xlim(min_val, max_val)

    ax.set_aspect('equal', 'box')

    ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='-', linewidth=1)

    if num_ticks is not None:
        ax.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
        ax.yaxis.set_major_locator(plt.MaxNLocator(num_ticks))

    # Make the ticks go into and out of the plot and make them larger
    ax.tick_params(axis='both', which='major', direction='inout', length=10, width=1)

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
        cbar.set_label('Frequency')

    else:
        ax.scatter(x, y, alpha=alpha, s=s, **kwargs)

    return ax
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