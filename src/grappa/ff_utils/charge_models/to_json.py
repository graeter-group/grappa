#%%
if __name__=="__main__":
    import numpy as np
    from pathlib import Path
    import json
    import os
    import copy

    #%%
    old_path = Path(__file__).parent/Path("npy_charge_dicts")
    new_path = Path(__file__).parent/Path("charge_dicts")
    for p in old_path.rglob("*.npy"):
        d = np.load(p, allow_pickle=True)
        d = d.item()

        # rename all keys (ie residues) to uppercase
        old_keys = list(d.keys())
        for k in old_keys:
            d[k.upper()] = d.pop(k)

        # convert all values to floats
        for k in d.keys():
            for k2 in d[k].keys():
                try:
                    d[k][k2] = float(d[k][k2])
                except:
                    for k3 in d[k][k2].keys():
                        try:
                            d[k][k2][k3] = float(d[k][k2][k3])
                        except:
                            pass


                        
        # hard code some symmetries into the radical dictionaries:
        is_rad = "_rad" in p.stem
        if is_rad:

            # first entry: first is missing, second is to be copied from
            # then the replacements, second entry is to be copied from
            copies = {
                "PHE": (["CD1", "CD2"],
                        [
                            ["HD2", "HD1"],
                        ]),
                "TYR": (["CE1", "CE2"],
                        [
                            ["HE2", "HE1"],
                        ]),
            }
            for res in copies.keys():
                new = copies[res][0][0]
                old = copies[res][0][1]
                replacement_list = copies[res][1]
                
                # copy the whole entry:
                d[res][new] = copy.deepcopy(d[res][old])

                # swap the role of the two atoms:
                d[res][new][new] = d[res][old][old]
                d[res][new][old] = d[res][old][new]
                
                # replace hydrogens that are not there anymore because of the swap:
                for new_, old_ in replacement_list:
                    try:
                        d[res][new][new_] = d[res][old][old_]
                    except:
                        print(f"old radical: {old}, new radical: {new},\n old_: {old_}, new_: {new_}\n keys: {d[res][old].keys()}\n")
                        raise
                    d[res][new].pop(old_)




        os.makedirs(str(new_path/Path(p.parent.stem)), exist_ok=True)

        with open(str(new_path/Path(p.parent.stem)/Path(p.stem))+".json", "w") as f:
            json.dump(d, f, indent=4, sort_keys=True)

#%%
