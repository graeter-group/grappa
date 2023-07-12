#%%
from pathlib import Path
spicepath = "/hits/fast/mbm/seutelf/data/datasets/SPICE-1.1.2.hdf5"
dipeppath = str(Path(spicepath).parent/Path("dipeptides_spice.hdf5"))
smallspicepath = str(Path(spicepath).parent/Path("small_spice.hdf5"))

#%%
if __name__ == "__main__":
    import random
    import os

    import h5py

    #%%
    # get keys that correspond to dipeptides:   
    def is_pep_string(x):
        try:
            float(x)
            return False
        except:
            if len(x) == 7:
                if x[3]=="-":
                    return True
            return False

    with h5py.File(spicepath, "r") as f:
        keylist = list(f.keys())
        filtered = filter(is_pep_string, keylist)
        names_ds = [entry for entry in filtered]
        print(len(names_ds))
    #%%
    # make hdf5 file only containing the dipeptides:
    if os.path.exists(dipeppath):
        os.remove(dipeppath)
    with h5py.File(dipeppath, "w") as f:
        with h5py.File(spicepath, "r") as read:
            for i, name in enumerate(names_ds):
                print(f"{i}/{len(names_ds)-1}: {name[:10]}", end='\r')
                grp = f.create_group(name)
                for key in read[name].keys():
                    f[name][key] = read[name][key][()]
            print()
            print("done")
    #%%
    # inspect dipeptide names:
    print(len(set(names_ds)), len(names_ds))
    caps = [name.upper() for name in names_ds]
    print(len(set(caps)), len(caps))
    #%%

    # make small spice containing only 20 molecules:

    if os.path.exists(smallspicepath):
        os.remove(smallspicepath)
    random.shuffle(names_ds)
    with h5py.File(smallspicepath, "w") as f:
        with h5py.File(dipeppath, "r") as read:
            for i, name in enumerate(names_ds[:20]):
                print(f"{i}/{len(names_ds[:20])-1}: {name[:10]}", end='\r')
                grp = f.create_group(name)
                for key in read[name].keys():
                    f[name][key] = read[name][key][()]
            print()
            print("done")

    import shutil
    shutil.copyfile(smallspicepath, Path(__file__).parent.parent/Path("small_spice/small_spice.hdf5"))
    #%%

    # %%
