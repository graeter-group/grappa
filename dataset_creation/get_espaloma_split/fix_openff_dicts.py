import json
from pathlib import Path

def fix_openff_dicts(input_prefix):
    # get all mol.json files, no matter how deep they are nested:

    molpaths = Path(input_prefix).glob('**/mol.json')

    for i,mol_path in enumerate(molpaths):
        with open(mol_path, 'r') as f:
            print(i, end='\r')
            moldata = json.load(f)
            # convert from str to dict:
            moldata = json.loads(moldata)
            if not 'partial_charge_unit' in moldata.keys():
                moldata['partial_charge_unit'] = moldata['partial_charges_unit']
            if "hierarchy_schemes" not in moldata.keys():
                moldata["hierarchy_schemes"] = dict()
            moldata = json.dumps(moldata)
        with open(mol_path, 'w') as f:
            json.dump(moldata, f)


input_prefix = Path(__file__).parent.parent.parent/'data/esp_data'
assert input_prefix.exists()

fix_openff_dicts(input_prefix)