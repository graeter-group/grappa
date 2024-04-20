"""
NOTE: CAN BE DELETED IN THE FUTURE, NOT NEEDED ANYMORE.
"""

import torch, dgl
from pathlib import Path
from grappa.constants import CHARGE_MODELS

CLASSICAL_CHARGES = ["AA_radical", "Capped_AA_opt_rad", "Capped_AA_rad", "Capped_AA_scan_rad", "dipeptide_rad", "hyp-dop_amber99sbildn", "AA_bondbreak_rad_amber99sbildn"]

CLASSICAL_CHARGE_TAG = ['amber99']

DATAPATH = Path(__file__).parent.parent/"data/dgl_datasets"

def add_charge_model_feat(dgl_graphlist, charge_model='am1BCC'):
    # one-hot encode the charge model:
    assert charge_model in CHARGE_MODELS, f"Charge model {charge_model} not in {CHARGE_MODELS}."
    v = [cm == charge_model for cm in CHARGE_MODELS]

    v = torch.tensor(v).float()

    for i, g in enumerate(dgl_graphlist):

        dgl_graphlist[i].nodes['n1'].data['charge_model'] = v.repeat(g.number_of_nodes('n1'), 1)

    return dgl_graphlist

def process_dspath(dspath, overwrite=True):
    dspath = Path(dspath)

    # load dataset/graphs.bin:
    dgl_graphlist = dgl.load_graphs(str(dspath/"graphs.bin"))[0]

    if dspath.name in CLASSICAL_CHARGES or any([tag in dspath.name for tag in CLASSICAL_CHARGE_TAG]):
        charge_model = "amber99"
    else:
        charge_model = "am1BCC"

    if not overwrite:
        if "charge_model" in dgl_graphlist[0].nodes["n1"].data.keys():
            print(f"    Skipping {dspath.name}... Already has charge_model feature.")
            return

    print(f"Processing {dspath.name}... Will receive charge model feat: {charge_model}")

    dgl_graphlist = add_charge_model_feat(dgl_graphlist, charge_model=charge_model)

    # save dataset/graphs.bin:
    dgl.save_graphs(str(dspath/"graphs.bin"), dgl_graphlist)

    return

def main(overwrite=True):
    for dspath in DATAPATH.iterdir():
        if dspath.is_dir():
            if (dspath/"graphs.bin").exists():
                process_dspath(dspath, overwrite=overwrite)

    return


if __name__ == "__main__":
    main(overwrite=True)

    import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-o', "--overwrite", action="store_true", help="Overwrite existing has_alternative_charge flag.")

    # args = parser.parse_args()

    # main(overwrite=args.overwrite)