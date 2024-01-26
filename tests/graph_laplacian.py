#%%
import dgl
import torch
import numpy as np
import scipy.sparse as sp
import networkx as nx

def laplacian_positional_encoding(dgl_graph, k):
    """
    Compute the first k non-trivial eigenvectors of the graph Laplacian
    and use them as node embeddings.

    Args:
    dgl_graph (dgl.DGLGraph): The DGL graph.
    k (int): The number of eigenvectors to compute.

    Returns:
    torch.Tensor: A tensor containing the first k eigenvectors.
    """
    # Convert DGL graph to NetworkX graph for easier manipulation
    homgraph = dgl.node_type_subgraph(dgl_graph, ['n1'])
    xyz = homgraph.ndata['xyz']
    
    # make sure that the nodes have the correct order:
    assert (xyz == dgl_graph.nodes['n1'].data['xyz']).all()
    
    nx_graph = homgraph.to_networkx().to_undirected()

    # Compute the normalized Laplacian matrix
    laplacian = nx.normalized_laplacian_matrix(nx_graph)
    laplacian = sp.csr_matrix(laplacian)

    # Compute the first k+1 smallest eigenvalues and eigenvectors
    # (k+1 because we discard the first trivial eigenvector)
    eigenvalues, eigenvectors = sp.linalg.eigsh(laplacian, k=k+1, which='SM')

    # Ignore the first eigenvector (trivial, corresponding to the eigenvalue 0)
    eigenvectors = eigenvectors[:, 1:]

    # Convert to torch tensor
    embeddings = torch.from_numpy(eigenvectors).float()

    return embeddings
#%%
from grappa.data import Dataset

ds = Dataset.from_tag('spice-des-monomers')

g = ds[0][0]

# Example usage
# Assuming you have a DGL graph object `g`
k = 5  # Number of eigenvectors to use
import time
start = time.time()
embeddings = laplacian_positional_encoding(g, k)
print(f'elapsed time in ms: {(time.time()-start)*1000}')
# %%
embeddings
# %%
from grappa.data import GraphDataLoader
import copy
from grappa.utils import dgl_utils

loader = GraphDataLoader(ds, batch_size=2)
g, _ = next(iter(loader))
graph_list = copy.deepcopy(dgl_utils.unbatch(g))

embeddings1 = laplacian_positional_encoding(g, k)
g.nodes['n1'].data['laplacian_pos_enc'] = embeddings1

embeddings2 = [laplacian_positional_encoding(g_, k) for g_ in graph_list]
for i, g_ in enumerate(graph_list):
    g_.nodes['n1'].data['laplacian_pos_enc'] = embeddings2[i]

g = dgl_utils.unbatch(g)

assert all([torch.allclose(g_.nodes['n1'].data['laplacian_pos_enc'], g[i].nodes['n1'].data['laplacian_pos_enc']) for i, g_ in enumerate(graph_list)])
# %%
print(embeddings1)
print(embeddings2)
print('batching changes embeddings!')
# %%
import dgl
import torch
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

def laplacian_positional_encoding2(dgl_graph, k):
    """
    Compute the first k non-trivial eigenvectors of the graph Laplacian
    and use them as node embeddings.

    Args:
    dgl_graph (dgl.DGLGraph): The DGL graph.
    k (int): The number of eigenvectors to compute.

    Returns:
    torch.Tensor: A tensor containing the first k eigenvectors.
    """
    # Extract adjacency matrix as a sparse matrix
    adj_matrix = dgl_graph.adjacency_matrix().to_scipy('csr')

    # Degree matrix as a diagonal matrix
    degrees = adj_matrix.sum(axis=1).A.flatten()
    deg_matrix = sp.diags(degrees)

    # Compute the normalized Laplacian matrix: L = D^(-1/2) * (D - A) * D^(-1/2)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
    laplacian = D_inv_sqrt @ (deg_matrix - adj_matrix) @ D_inv_sqrt

    # Compute the first k+1 smallest eigenvalues and eigenvectors
    eigenvalues, eigenvectors = sp.linalg.eigsh(laplacian, k=k+1, which='SM')

    # Ignore the first eigenvector (trivial, corresponding to the eigenvalue 0)
    eigenvectors = eigenvectors[:, 1:]

    # Convert to torch tensor
    embeddings = torch.from_numpy(eigenvectors).float()

    return embeddings

# %%
g = ds[10][0]

assert laplacian_positional_encoding(g, 5).allclose(laplacian_positional_encoding2(g, 5))
# %%
