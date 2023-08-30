#%%
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from scipy.optimize import curve_fit

def affine_fit(x, a, b):
    return a * x + b

def lc_plot(data:Dict, title="Learning Curve", fontname="Arial", ylabel="Force RMSE [kcal/mol/Å]", 
            show=False, plotpath=None, fit=True, logx=False, logy=False, fontsize=16, connect_dots=False, ignore_n_worst=0, ylim=None, n_min=20, ignore_ns:List[int]=[], hlines:Dict=None, suffix=""):
    """
    data[ds_type][n_mols] = rmse_list
    """
    if fontname is not None:
        import matplotlib as mpl
        mpl.rc('font', family=fontname, size=fontsize)
    
    assert not plotpath is None or show, "Either plotpath or show must be True"

    if not plotpath is None:
        Path(plotpath).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    # select the first len(list(data.keys())) colors from the default color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(list(data.keys()))]

    # add a grid:
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    def is_int(s:str):
        try:
            int(s)
            return True
        except:
            return False

    if is_int(list(data.keys())[0]):
        data = {"": data}

    for i, (ds_type, ds_data) in enumerate(data.items()):
        keys_ = [int(k) for k in ds_data.keys() if int(k) >= n_min]
        keys_ = [k for k in keys_ if not k in ignore_ns]
        keys_.sort()
        
        x = np.array([float(k) for k in keys_])
        errs = []
        errs_std = []
        for k in keys_:
            errs_k = ds_data[str(k)]
            # sort errs and ignore the worst n
            if ignore_n_worst > 0 and len(errs_k) > ignore_n_worst:
                errs_k = np.sort(errs_k)[:-ignore_n_worst]
            errs.append(np.mean(errs_k))
            errs_std.append(np.std(errs_k))

        y_means = np.array(errs)
        y_stds = np.array(errs_std)        

        plt.errorbar(x, y_means, yerr=y_stds, linestyle='None' if not connect_dots else '-', marker='o', label=f"{ds_type}", color=colors[i], capsize=5)

        if not hlines is None:
            if ds_type in hlines.keys():
                plt.hlines(hlines[ds_type], min(x), max(x), color=colors[i], linestyle='--', alpha=1)

        if fit and logx and logy:
            popt, _ = curve_fit(affine_fit, np.log(x), np.log(y_means))
            y_fit = np.exp(affine_fit(np.log(x), *popt))
            plt.plot(x, y_fit, '--', color=colors[i])

    plt.title(title)
    plt.xlabel("Molecules in Training Set")
    plt.ylabel(ylabel)
    if not ylim is None:
        plt.ylim(ylim)

    if logx:
        plt.xscale('log')

    if logy:
        plt.yscale('log')

    plt.legend()

    if plotpath is not None:
        plt.savefig(str(Path(plotpath)/f"learning_curve{suffix}.png"), dpi=500)

    if show:
        plt.show()


#%%

if __name__ == "__main__":
    import json

#     with open("lc_grappa/Spice Dipeptides_lc_data.json", "r") as f:
#         data = json.load(f)
    
#     lc_plot(data, title="Learning Curve", fontname="Arial", ylabel="Force RMSE [kcal/mol/Å]", show=True, plotpath=None, fit=False, logx=True, logy=True, ignore_n_worst=1, connect_dots=True)

#     # %%

#     with open("lc_grappa/lc_data.json", "r") as f:
#         data = json.load(f)
    
#     lc_plot(data, title="Learning Curve", fontname="Arial", ylabel="Force RMSE [kcal/mol/Å]", show=True, plotpath=None, fit=False, logx=True, logy=True, ignore_n_worst=1, connect_dots=True)
# # %%

    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Plot learning curve')
    parser.add_argument('file', type=str, help='Path to json file with data')
    args = parser.parse_args()

    with open(args.file, "r") as f:
        data = json.load(f)
    
    lc_plot(data, title="Learning Curve", fontname="Arial", ylabel="Force RMSE [kcal/mol/Å]", show=True, plotpath=str(Path(args.file).parent/Path(args.file).stem), fit=False, logx=True, logy=False, ignore_n_worst=1, connect_dots=True)