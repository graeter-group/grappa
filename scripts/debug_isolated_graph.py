#%%
from grappa.data import GraphDataLoader, Dataset
import dgl
from tqdm import tqdm
#%%

ds = Dataset.from_tag("spice-pubchem")
#%%
loader = GraphDataLoader(ds, batch_size=1, shuffle=False, conf_strategy=32)
# %%
for g,_ in tqdm(loader):
    g = g.to('cuda')
    try:
        g_ = g.node_type_subgraph(["n1"])
        g_ = dgl.to_homogeneous(g_)
    except Exception as e:
        print(g.num_nodes('n1'))
        raise

# %%
