"""
NOTE: CAN BE DELETED IN THE FUTURE, NOT NEEDED ANYMORE.
"""

import torch, dgl
from pathlib import Path

DATAPATH = Path(__file__).parent.parent/"data/dgl_datasets"

def process_dspath(dspath):
    dspath = Path(dspath)
    if 'rad' in dspath.name:
        return

    # load dataset/graphs.bin:
    dgl_graphlist = dgl.load_graphs(str(dspath/"graphs.bin"))[0]

    print(f"Processing {dspath.name}...")

    for i, g in enumerate(dgl_graphlist):

        if not 'is_radical' in dgl_graphlist[i].nodes['n1'].data.keys():
            dgl_graphlist[i].nodes['n1'].data['is_radical'] = torch.zeros(g.number_of_nodes('n1'))

    # save dataset/graphs.bin:
    dgl.save_graphs(str(dspath/"graphs.bin"), dgl_graphlist)

    return

def main():
    for dspath in DATAPATH.iterdir():
        if dspath.is_dir():
            if (dspath/"graphs.bin").exists():
                process_dspath(dspath)

    return


if __name__ == "__main__":
    main()


    # parser = argparse.ArgumentParser()
    # parser.add_argument('-o', "--overwrite", action="store_true", help="Overwrite existing has_alternative_charge flag.")

    # args = parser.parse_args()

    # main(overwrite=args.overwrite)