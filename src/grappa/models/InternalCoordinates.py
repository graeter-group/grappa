import dgl
import torch

class InternalCoordinates(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(InternalCoordinates, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, g):
        return internal_coordinates(g, *self.args, **self.kwargs)
    
# inspired by the espaloma function geometry_in_graph

def internal_coordinates(g):
    """
    Assign values to geometric entities in graphs.

    Parameters
    ----------
    g : `dgl.DGLHeteroGraph`
        Input graph.

    Returns
    -------
    g : `dgl.DGLHeteroGraph`
        Output graph.

    Notes
    -----
    This function modifies graphs in-place.

    """

    #==========================================================================
    if not "x" in g.nodes["n2"].data.keys():
        # bonds:
        pairs = g.nodes["n2"].data["idxs"]
        # this has shape num_pairs, 2

        positions = g.nodes["n1"].data["xyz"][pairs]
        # this has shape num_pairs, 2, num_confs, 3

        # compute bond length
        x = distance(positions[:, 0, :, :], positions[:, 1, :, :])
        # this has shape num_pairs, num_confs

        # write this in the graph:
        g.nodes["n2"].data["x"] = x


    #==========================================================================
    if "n3" in g.ntypes:
        if not "x" in g.nodes["n3"].data.keys():

            # compute bond angle
            pairs = g.nodes["n3"].data["idxs"]
            # this has shape num_pairs, 3

            if len(pairs) == 0:
                x = torch.zeros((0, g.nodes["n1"].data["xyz"].shape[1]), device=g.nodes["n1"].data["xyz"].device)

            else:
                positions = g.nodes["n1"].data["xyz"][pairs]
                # this has shape num_pairs, 3, num_confs, 3

                x = angle(positions[:, 0, :, :], positions[:, 1, :, :], positions[:, 2, :, :])
                # this has shape num_pairs, num_confs

            # write this in the graph:
            g.nodes["n3"].data["x"] = x


    #==========================================================================
    # compute torsion angle
    improper_needed = "n4_improper" in g.ntypes
    improper_needed = (not "x" in g.nodes["n4_improper"].data.keys()) if improper_needed else False

    pairs = None
    if "n4" in g.ntypes:
        if not "x" in g.nodes["n4"].data.keys():

            pairs = g.nodes["n4"].data["idxs"]

            if "n4_improper" in g.ntypes:
                # we can treat proper and improper the same way:
                pairs = torch.cat((pairs, g.nodes["n4_improper"].data["idxs"]), dim=0)

    elif improper_needed:

        pairs = g.nodes["n4_improper"].data["idxs"]
        # this has shape num_pairs, 4

    if not pairs is None:

        if len(pairs) == 0:
            x = torch.zeros((0, g.nodes["n1"].data["xyz"].shape[1]), device=g.nodes["n1"].data["xyz"].device)
        else:

            positions = g.nodes["n1"].data["xyz"][pairs]
            # this has shape num_pairs, 4, num_confs, 3

            try:
                x = dihedral(positions[:, 0, :, :], positions[:, 1, :, :], positions[:, 2, :, :], positions[:, 3, :, :])
                # this has shape num_pairs, num_confs
            except IndexError:
                print(pairs.shape)
                print(g.nodes["n4"].data["idxs"].shape)
                print(g.nodes["n1"].data["xyz"].shape)
                raise


        # write this in the graph:
        if "n4_improper" in g.ntypes:
            if not "n4" in g.ntypes:
                num_propers = 0
            else:
                num_propers = g.nodes["n4"].data["idxs"].shape[0]
                g.nodes["n4"].data["x"] = x[:num_propers]
            g.nodes["n4_improper"].data["x"] = x[num_propers:]
        else:
            g.nodes["n4"].data["x"] = x
    

    return g

# from espaloma:

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

# =============================================================================
# IMPORTS
# =============================================================================



# =============================================================================
# SINGLE GEOMETRY ENTITY
# =============================================================================
def distance(x0, x1):
    """ Distance. """
    return torch.norm(x0 - x1, p=2, dim=-1)


def _angle(r0, r1):
    """ Angle between vectors. """

    angle = torch.atan2(
        torch.norm(torch.cross(r0, r1), p=2, dim=-1),
        torch.sum(torch.mul(r0, r1), dim=-1),
    )

    return angle


def angle(x0, x1, x2):
    """ Angle between three points. """
    left = x1 - x0
    right = x1 - x2
    return _angle(left, right)


def _dihedral(r0, r1):
    """ Dihedral between normal vectors. """
    return _angle(r0, r1)


def dihedral(
    x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
) -> torch.Tensor:
    """
    Dihedral between four points, assuming that x1 is the central atom.

    Reference
    ---------
    Closely follows implementation in Yutong Zhao's timemachine:
        https://github.com/proteneer/timemachine/blob/1a0ab45e605dc1e28c44ea90f38cb0dedce5c4db/timemachine/potentials/bonded.py#L152-L199
    """
    # check input shapes

    assert x0.shape == x1.shape == x2.shape == x3.shape

    # compute displacements 0->1, 2->1, 2->3
    r01 = x1 - x0 + torch.randn_like(x0) * 1e-5
    r21 = x1 - x2 + torch.randn_like(x0) * 1e-5
    r23 = x3 - x2 + torch.randn_like(x0) * 1e-5

    # compute normal planes
    n1 = torch.cross(r01, r21)
    n2 = torch.cross(r21, r23)

    rkj_normed = r21 / torch.norm(r21, dim=-1, keepdim=True)

    y = torch.sum(torch.mul(torch.cross(n1, n2), rkj_normed), dim=-1)
    x = torch.sum(torch.mul(n1, n2), dim=-1)

    # choose quadrant correctly
    theta = torch.atan2(y, x)

    return theta
