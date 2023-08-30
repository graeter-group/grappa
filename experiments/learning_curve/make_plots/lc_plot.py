#%%
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from scipy.optimize import curve_fit

def affine_fit(x, a, b):
    return a * x + b

def lc_plot(data:Dict, title="Learning Curve", fontname="Arial", ylabel="Force RMSE [kcal/mol/Å]", 
            show=False, plotpath=None, fit=True, logx=False, logy=False, fontsize=16, connect_dots=False, ignore_n_worst=0):
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
    # plt.grid(True, axis='y', linestyle='--', alpha=1)

    def is_int(s:str):
        try:
            int(s)
            return True
        except:
            return False

    if is_int(list(data.keys())[0]):
        data = {"": data}

    for i, (ds_type, ds_data) in enumerate(data.items()):
        keys_ = sorted(list(ds_data.keys()), key=float)
        
        if ignore_n_worst > 0:
            if len(keys_) > ignore_n_worst:
                keys_ = keys_[:-ignore_n_worst]
        
        x = np.array([float(k) for k in keys_])
        y_means = np.array([np.mean(ds_data[k]) for k in keys_])
        y_stds = np.array([np.std(ds_data[k]) for k in keys_])

        plt.errorbar(x, y_means, yerr=y_stds, linestyle='None' if not connect_dots else '-', marker='o', label=f"{ds_type}", color=colors[i], capsize=5)

        if fit and logx and logy:
            popt, _ = curve_fit(affine_fit, np.log(x), np.log(y_means))
            y_fit = np.exp(affine_fit(np.log(x), *popt))
            plt.plot(x, y_fit, '--', color=colors[i])

    plt.title(title)
    plt.xlabel("Molecules in Training Set")
    plt.ylabel(ylabel)

    if logx:
        plt.xscale('log')

    if logy:
        plt.yscale('log')

    plt.legend()

    if plotpath is not None:
        plt.savefig(str(Path(plotpath)/"learning_curve.png"), dpi=500)

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
    
    lc_plot(data, title="Learning Curve", fontname="Arial", ylabel="Force RMSE [kcal/mol/Å]", show=True, plotpath=str(Path(args.file).parent/Path(args.file).stem), fit=False, logx=True, logy=True, ignore_n_worst=1, connect_dots=True)