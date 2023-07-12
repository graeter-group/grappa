#%%
from pathlib import Path

# import sys
# module_path = str(Path(__file__).parent.parent.parent.parent)
# if module_path not in sys.path:
#     sys.path.append(module_path)

from grappa.PDBData.xyz2res.model_ import Representation
from dgl import load_graphs
from dgl.data.utils import split_dataset
import torch



if __name__ == "__main__":

    model = Representation(256,out_feats=1,n_residuals=2,n_conv=1)

    graphs, _ = load_graphs("./data/pdbs_dgl.bin")

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)

    ds_tr, ds_vl, ds_te = split_dataset(graphs, [0.8,0.1,0.1], shuffle=True, random_state=0)

    #%%

    def predict(scores):
        return torch.where(scores >=0, 1, 0)

    def accuracy(pred, ref):
        t = torch.where(pred==ref, 1, 0)
        acc = t.sum() / torch.ones_like(t).sum()
        return acc.item()

    #%%
    epochs = 15
    model = model.to("cuda")
    for epoch in range(epochs):
        tl = 0
        preds = []
        refs = []
        # train loop:
        for g in ds_tr:
            optim.zero_grad()
            g = g.to("cuda")
            g = model(g)
            pred = g.ndata["h"][:,0]
            ref = g.ndata["c_alpha"].float()
            loss = loss_fn(pred, ref)
            loss.backward()
            optim.step()
            preds.append(predict(pred))
            refs.append(ref)
            tl += loss
        preds = torch.cat(preds,dim=0)
        refs = torch.cat(refs,dim=0)
        info = f"train loss: {tl.item():.4f}, train acc: {accuracy(preds, refs):.4f}"


        # val accuracy:
        preds = []
        refs = []
        # train loop:
        for g in ds_vl:
            g = g.to("cuda")
            g = model(g)
            pred = g.ndata["h"][:,0]
            ref = g.ndata["c_alpha"].float()
            preds.append(predict(pred))
            refs.append(ref)
        preds = torch.cat(preds,dim=0)
        refs = torch.cat(refs,dim=0)
        info += f", val acc: {accuracy(preds, refs):.4f}"

        print(info)
    # %%
    model.eval()

    # test accuracy:
    preds = []
    refs = []
    # train loop:
    for g in ds_te:
        g = g.to("cuda")
        g = model(g)
        pred = g.ndata["h"][:,0]
        ref = g.ndata["c_alpha"].float()
        preds.append(predict(pred))
        refs.append(ref)
    preds = torch.cat(preds,dim=0)
    refs = torch.cat(refs,dim=0)
    info = f"final test acc: {accuracy(preds, refs):.4f}"
    print(info)

    # save the model
    model = model.to("cpu")
    torch.save(model.state_dict(), Path(__file__).parent.parent / Path("match_model2.pt"))
    # %%
