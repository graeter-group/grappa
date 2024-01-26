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
print('total number of molecules in the espaloma dataset\nthis should coincide with the number of molecules in the espaloma paper!')
# %%
# now the same for the converted dataset directly:

grappa_num_dict = {}
from grappa.data import MolData

for dir in (this_dir.parent/'grappa_datasets').iterdir():
    if not dir.is_dir():
        continue
    # count number of files
    num_files = len(list(dir.iterdir()))
    grappa_num_dict[dir.name] = num_files

print(json.dumps(grappa_num_dict, indent=4))
print('total number of molecules in the grappa dataset')
differences = {key: num_dict[key] - grappa_num_dict[key] for key in num_dict}
if any(differences.values()):
    print('the numbers of molecules are not the same. differences:')
    print(json.dumps(differences, indent=4))
print('the numbers of molecules are the same.')
# %%
