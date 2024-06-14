#%%
from grappa.data import Dataset
from grappa.utils.graph_utils import get_isomorphisms
#%%
ds = Dataset.from_tag("dipeptides-300K-amber99")
validate_graphs = [ds.graphs[i] for i in range(10)]
validate_graphs = validate_graphs + [ds.graphs[i] for i in range(2)]

# %%
assert get_isomorphisms(validate_graphs) == {(0, 10), (1, 11)}
# %%
sum(get_isomorphisms(ds.graphs))
# %%
spice_dataset = Dataset.from_tag("spice-dipeptide")
len(get_isomorphisms(spice_dataset.graphs))
# %%
# now find one-to one correspondences between the two datasets:
isomorphisms = get_isomorphisms(ds.graphs, spice_dataset.graphs)
len(isomorphisms)
# %%
