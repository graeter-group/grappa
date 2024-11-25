import dgl, networkx as nx, numpy as np, torch
from typing import List, Tuple, Set
import matplotlib.pyplot as plt
from grappa.utils.graph_utils import as_nx


def draw_mol(nx_graph: nx.Graph, with_idx: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw a NetworkX graph representing a molecule.
    
    Parameters:
    - nx_graph (nx.Graph): A NetworkX graph object representing a molecule. 
                           Assumes that the node attribute 'atomic_number' is present.
    - with_idx (bool): A flag indicating whether to display node indices.
    """
    # Define atomic colors (sulfur yellow:)
    COLORS = {1: 'white', 6: 'black', 7: 'blue', 8: 'red', 16: 'yellow'}

    # Prepare node labels and colors
    labels = {}
    colors = []
    for node, data in nx_graph.nodes(data=True):
        atomic_number = data.get('atomic_number', None)
        if atomic_number is not None:
            atomic_number = int(atomic_number)
        
        # Set node color based on atomic number
        color = COLORS.get(atomic_number, 'gray')
        colors.append(color)
        
        # Set node label
        if with_idx:
            labels[node] = str(node)
        else:
            labels[node] = data.get('element', str(node))
    
    # Adjust figure size based on the number of nodes
    size = np.sqrt(len(nx_graph.nodes())) / np.sqrt(40) * 8
    fig, ax = plt.subplots(figsize=(size, size))

    # Improved layout to avoid node overlap
    pos = nx.spring_layout(nx_graph, seed=0, k=0.1)  # Higher `k` for better spacing

    nx.draw_networkx_edges(
        nx_graph, pos, ax=ax, width=2.5, alpha=0.7, arrowstyle='-'
    )

    # Draw nodes with a black border
    nx.draw_networkx_nodes(
        nx_graph, pos, ax=ax, node_color=colors, node_size=800,
        edgecolors='black', linewidths=1.2, alpha=0.7
    )


    # Draw labels
    nx.draw_networkx_labels(
        nx_graph, pos, labels=labels, ax=ax, font_size=10,
        font_color='black', font_weight='bold'
    )
    
    ax.set_axis_off()
    plt.tight_layout()
    return fig, ax

def draw_dgl_graph(g: dgl.DGLGraph, with_idx: bool = True) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw a DGL graph representing a molecule.
    
    Parameters:
    - g (dgl.DGLGraph): A DGL graph object representing a molecule. 
    - with_idx (bool): A flag indicating whether to display node indices.
    """
    # Convert DGL graph to NetworkX
    nx_graph = as_nx(g)
    return draw_mol(nx_graph, with_idx=with_idx)