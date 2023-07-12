#%%

import dgl
import torch
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# features must be of shape n_nodes x 1 to be plotted.
def draw_mol(graph, arrows=False, show=True, color_dict={"H":"grey", "C":"black", "O":"red", "N":"darkblue", "P":"orange", "S":"yellow"}, alpha=0.5, features=[], labels=None, title="molecular graph"):

    elems = graph.ndata["atomic_number"]
    elems = torch.argmax(elems, dim=-1)

    cmap = plt.get_cmap("tab10")
    i = 0
    nodes = {}
    if len(graph.ntypes)>1:
        hom = dgl.to_homogeneous(graph)
        G = dgl.to_networkx(hom)
    else:
        hom = graph
        G = dgl.to_networkx(hom, node_attrs=features)
    
    for n, ntype in [(1,"H"), (6,"C"), (7,"N"), (8,"O"), (15,"P"), (16,"S")]:
        #nodes[ntype] = torch.argwhere(hom.ndata["_TYPE"]==n_num)[:,0].tolist()
        nodes[ntype] = torch.argwhere(elems==n)[:,0].tolist()
        if not ntype in color_dict.keys():
            color_dict[ntype] = np.array([cmap(i)])
            i+=1

    plt.figure(figsize=[10,6])
    plt.title(title)

    # pos = nx.spring_layout(G, seed=seed, k=0.05, iterations=20)
    pos = nx.fruchterman_reingold_layout(G)

    for ntype in nodes.keys():
        nx.draw_networkx_nodes(G, pos, nodelist=nodes[ntype], node_color=color_dict[ntype], node_size=200, alpha=alpha)

    nx.draw_networkx_edges(G, pos, edge_color="gray", width=2, alpha=0.3, arrows=arrows)
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="black")


    if show:
        plt.show()

# %%
