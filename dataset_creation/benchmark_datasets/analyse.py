#%%
from pathlib import Path
import json

#%%
this_dir = Path(__file__).parent

# go to esp dir:
this_dir = this_dir.parent.parent/ 'data/esp_data'

num_dict = {}

for dir in this_dir.iterdir():
    if not dir.is_dir():
        continue
    if 'duplicate' in dir.name:
        continue
    # count number of files
    num_files = len(list(dir.iterdir()))
    num_dict[dir.name] = num_files

print(json.dumps(num_dict, indent=4))
#%%
# now analyse the duplicates
for dir in this_dir.iterdir():
    if not dir.is_dir():
        continue
    if 'duplicate' not in dir.name:
        continue
    # step one deeper (iterate over 1/ 2/ 3/ ...)
    for subdir in dir.iterdir():
        if not subdir.is_dir():
            continue
        # (iterate over 32/ds1, 32/ds2, ...)
        for subsubdir in subdir.iterdir():
            if not subsubdir.is_dir():
                continue
            # count number of files
            num_mols_here = 1
            num_dict[subsubdir.name] += num_mols_here

print(json.dumps(num_dict, indent=4))
print('this should coincide with the number of molecules in the espaloma paper!')
# %%
# now the same for the converted dataset directly:

grappa_num_dict = {}

for dir in (this_dir.parent/'datasets').iterdir():
    if not dir.is_dir():
        continue
    # count number of files
    num_files = len(list(dir.iterdir()))
    grappa_num_dict[dir.name] = num_files

print(json.dumps(grappa_num_dict, indent=4))
assert grappa_num_dict == num_dict
# %%
