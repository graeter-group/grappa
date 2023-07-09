


# MIT License

# Copyright (c) 2020 Yuanqing Wang @ choderalab // MSKCC

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch


def get_model():
    """
    Returns a default espaloma model (https://espaloma.wangyq.net/experiments/qm_fitting.html) for comparision. In the espaloma code, improper torsion were not considered yet.
    """
    import espaloma as esp
    representation = esp.nn.Sequential(
        feature_units=117,
        input_units=128,
        layer=esp.nn.layers.dgl_legacy.gn("SAGEConv"), # use SAGEConv implementation in DGL
        config=[128, "relu", 128, "relu", 128, "relu"], # 3 layers, 128 units, ReLU activation
        )

    readout = esp.nn.readout.janossy.JanossyPooling(
        in_features=128, config=[128, "relu", 128, "relu", 128, "relu"],
        out_features={              # define modular MM parameters Espaloma will assign
            2: {"log_coefficients": 2}, # bond linear combination, enforce positive
            3: {"log_coefficients": 2}, # angle linear combination, enforce positive
            4: {"k": 6}, # torsion barrier heights (can be positive or negative)
        },
    )

    class set_improper_zero(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, g):
            # for l in ["n2", "n3"]:
            #     for i, p in enumerate(["k", "eq"]):
            #         g.nodes[l].data[p] = g.nodes[l].data["coefficients"][:, i].unsqueeze(dim=-1)

            g.nodes["n4_improper"].data["k"] = g.nodes["n4_improper"].data["k_amber99sbildn"]
            return g

    model = torch.nn.Sequential(
        representation, readout, esp.nn.readout.janossy.ExpCoefficients(),
        esp.nn.readout.janossy.LinearMixtureToOriginal(),
        set_improper_zero(),
    )
    
    return model