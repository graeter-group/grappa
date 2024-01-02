#%%
from grappa.data import Dataset
from grappa.utils.dataset_utils import get_data_path
import json
from pathlib import Path

#%%


datasets = [
        str(get_data_path()/"dgl_datasets"/dsname) for dsname in
        [
            "protein-torsion",
            'pepconf-dlc',
        ]
    ]

ds = Dataset()
for d in datasets:
    ds += Dataset.load(d).slice(0,50)

#%%

normal_split_ids = ds.calc_split_ids(seed=42, partition=[0.8, 0.1, 0.1])

if not Path('split_ids.json').exists():
    with open('split_ids.json', 'w') as f:
        json.dump(normal_split_ids, f, indent=4)
else:
    with open('split_ids.json', 'r') as f:
        assert normal_split_ids == json.load(f)
    print('normal splits are indeed the same as before')

#%%
# check whether there are overlaps between the splits
ids = ds.mol_ids
out = normal_split_ids

assert len(out['train']) + len(out['val']) + len(out['test']) == len(set(list(ids))), f"Split failed, {len(out['train']) + len(out['val']) + len(out['test'])} != {len(ids)}"

assert len(set(out['train']).intersection(set(out['val']))) == 0, "Train and val sets must not overlap"
assert len(set(out['train']).intersection(set(out['test']))) == 0, "Train and test sets must not overlap"
assert len(set(out['val']).intersection(set(out['test']))) == 0, "Val and test sets must not overlap"

print('No overlaps between splits')
# %%

# now for the k-fold splits:

from grappa.utils.torch_utils import get_k_fold_split_ids
# %%
ids = ds.mol_ids
ds_names = ds.subdataset

k_fold_split_ids = get_k_fold_split_ids(ids, ds_names, k=5, seed=42)
if not Path('k_fold_split_ids.json').exists():
    with open('k_fold_split_ids.json', 'w') as f:
        json.dump(k_fold_split_ids, f, indent=4)
else:
    with open('k_fold_split_ids.json', 'r') as f:
        assert k_fold_split_ids == json.load(f)
    print('k_fold splits are indeed the same as before')
# %%
# do the same tests as above:

# first, check whether each fold has disjoint tr, vl, te:
for i in range(5):
    
    assert len(k_fold_split_ids[i]['train']) + len(k_fold_split_ids[i]['val']) + len(k_fold_split_ids[i]['test']) == len(set(list(ids))), f"Split failed, {len(k_fold_split_ids[i]['train']) + len(k_fold_split_ids[i]['val']) + len(k_fold_split_ids[i]['test'])} != {len(ids)}"

    assert len(set(k_fold_split_ids[i]['train']).intersection(set(k_fold_split_ids[i]['val']))) == 0, "Train and val sets must not overlap"
    assert len(set(k_fold_split_ids[i]['train']).intersection(set(k_fold_split_ids[i]['test']))) == 0, "Train and test sets must not overlap"
    assert len(set(k_fold_split_ids[i]['val']).intersection(set(k_fold_split_ids[i]['test']))) == 0, "Val and test sets must not overlap"
# %%
# then, verify that the set of test splits is a disjoint partition of the ids:
test_ids = []
for i in range(5):
    test_ids += k_fold_split_ids[i]['test']

assert len(set(test_ids)) == len(test_ids), "Test ids must be unique"
assert set(test_ids) == set(ids), "Test ids must be a partition of the full set of ids"
# %%

# alternative way to do a (0.8,0.1,0.1) split:
other_normal_split_ids = get_k_fold_split_ids(ids, ds_names, k=10, seed=42, num_folds=5)[0]
# %%
# check whether there are overlaps between the splits
assert len(other_normal_split_ids['train']) + len(other_normal_split_ids['val']) + len(other_normal_split_ids['test']) == len(set(list(ids))), f"Split failed, {len(other_normal_split_ids['train']) + len(other_normal_split_ids['val']) + len(other_normal_split_ids['test'])} != {len(ids)}"

assert len(set(other_normal_split_ids['train']).intersection(set(other_normal_split_ids['val']))) == 0, "Train and val sets must not overlap"

assert len(set(other_normal_split_ids['train']).intersection(set(other_normal_split_ids['test']))) == 0, "Train and test sets must not overlap"

assert len(set(other_normal_split_ids['val']).intersection(set(other_normal_split_ids['test']))) == 0, "Val and test sets must not overlap"

# %%
