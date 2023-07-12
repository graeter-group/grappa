#%%
import torch
import dgl

from .constants import MAX_ELEMENT

"""
Implements one residual layer consisting of 2 message passing steps with optional self interaction (linear, activation, linear on the node features with shared weights), and a skip connection, also with optional self interaction. No nonlinearity is put on beginning and end of the block.
"""
class ResidualGraphBlock(torch.nn.Module):
    def __init__(self, n_feat, *message_args, activation=torch.nn.ELU(), message_class=dgl.nn.pytorch.conv.SAGEConv, self_interaction=True):
        super().__init__()
        self.module_skip = torch.nn.Sequential(
            torch.nn.Linear(n_feat,n_feat),
            activation,
            torch.nn.Linear(n_feat,n_feat),
        )

        self.module1 = message_class(*message_args)
        self.module2 = message_class(*message_args)
        self.activation = activation

        if self_interaction:
            self.self_interaction = torch.nn.Sequential(
                torch.nn.Linear(n_feat,n_feat),
                self.activation,
                torch.nn.Linear(n_feat,n_feat),
            )
        else:
            self.self_interaction = torch.nn.ELU()

    def forward(self, g, h):
        h_skip = self.module_skip(h)
        h = self.activation(self.module1(g,h))
        h = self.activation(self.self_interaction(h))
        h = self.module2(g,h)
        return h_skip+h


"""
Implementing a stack of ResidualGraphBlocks followed by message passing layers without skip connection and one large skip connection skipping all message passing steps. Also implements linear layers with node-level-shared weights as first and final layer.
"""
class Representation(torch.nn.Module):
    def __init__(self, h_feats, out_feats, in_feats=MAX_ELEMENT, n_residuals=3, n_conv=2, large_skip=True):
        super().__init__()

        self.large_skip = large_skip

        self.pre_dense = torch.nn.Sequential(
            torch.nn.Linear(in_feats, h_feats),
        )

        self.post_dense = torch.nn.Sequential(
            torch.nn.Linear(h_feats, out_feats),
        )
        if self.large_skip:
            self.post_dense = torch.nn.Sequential(
                torch.nn.Linear(2*h_feats, out_feats),
            )

        self.blocks = torch.nn.ModuleList([
                ResidualGraphBlock(h_feats, h_feats, h_feats, "mean", message_class=dgl.nn.pytorch.conv.SAGEConv)
                for i in range(n_residuals)
            ])

        self.blocks.extend(
            [
            dgl.nn.pytorch.conv.SAGEConv(h_feats, h_feats, aggregator_type="mean")
                for i in range(n_conv)
            ]
        )


    def forward(self, g, in_feature=None):
        if in_feature is None:
            in_feature = g.ndata["atomic_number"]*1.

        h = torch.nn.functional.elu(self.pre_dense(in_feature))
        if self.large_skip:
            h_skip = torch.nn.functional.elu(h)

        for block in self.blocks:
            h = torch.nn.functional.elu(block(g,h))

        if self.large_skip:
            h = torch.cat((h,h_skip), dim=-1)
            
        h = self.post_dense(h)
        g.ndata["h"] = h
        return g
# %%
# model = Representation(256,out_feats=1)