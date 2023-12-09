from .graph_attention import GrappaGNN
from .interaction_parameters import WriteParameters
from typing import Union, List, Tuple, Dict
import torch
from grappa.utils.graph_utils import get_default_statistics

class GrappaModel(torch.nn.Module):
    """
    Implements a grappa model which combines a Graph Neural Network (GNN) for feature extraction 
    followed by a parameter writing phase for bonds, angles, and torsions.

    The GNN part processes the input graph to extract features, which are then used by the parameter writer 
    to assign molecular parameters such as bond lengths, angles, and torsion angles.

    ----------
    Initialization Arguments:
    ----------
        out_feats (int): Number of output features from the GNN, also used as graph node features for the parameter writer.
        in_feats (int): Number of input features for the GNN.
        in_feat_name (Union[str, List[str]]): Names of input features for the GNN.
        in_feat_dims (Dict[str, int]): Dictionary mapping feature names to their dimensions for the GNN.
        gnn_width (int): Number of hidden node features in the GNN. Defaults to `out_feats`.
        gnn_attentional_layers (int): Number of attentional layers in the GNN.
        gnn_convolutions (int): Number of convolutional layers in the GNN.
        gnn_attention_heads (int): Number of attention heads in each attention block of the GNN.
        gnn_dropout_attention (float): Dropout rate in attention layers of the GNN.
        gnn_dropout_initial (float): Dropout rate after the initial linear layer of the GNN.
        gnn_dropout_conv (float): Dropout rate in convolutional layers of the GNN.
        gnn_dropout_final (float): Dropout rate in the final output layer of the GNN.
        parameter_dropout (float): Dropout rate in the parameter writer.

        {bond, angle, proper, improper}_transformer_depth (int, optional): Depth of the transformer for each parameter writer.
        {bond, angle, proper, improper}_n_heads (int, optional): Number of attention heads for each parameter writer.
        {bond, angle, proper, improper}_transformer_width (int, optional): Hidden feature dimension of the transformer for each parameter writer.
        {bond, angle, proper, improper}_symmetriser_depth (int, optional): Number of layers in the symmetriser for each parameter writer.
        {bond, angle, proper, improper}_symmetriser_width (int, optional): Feature dimension of the symmetriser for each parameter writer.
        n_periodicity_proper (int, optional): Number of periodicity terms for proper torsions. Defaults to 6.
        n_periodicity_improper (int, optional): Number of periodicity terms for improper torsions. Defaults to 3.
        gated_torsion (bool, optional): Flag indicating whether to use gated torsions. Defaults to False.
        wrong_symmetry (bool, optional): Flag indicating whether to implement the wrong symmetry constraint for improper torsions. Defaults to False.
        positional_encoding (bool, optional): Flag indicating whether to use positional encoding in the parameter writer. Defaults to True.
        layer_norm (bool, optional): Flag indicating whether to use layer normalization in the parameter writer. Defaults to True.
        self_interaction (bool, optional): Flag indicating whether to use self-interaction in the GNN. Defaults to True.
        stat_dict (dict, optional): Dictionary containing statistics for parameter initialization.
    The GrappaModel is a composite model combining the feature extraction capabilities of GrappaGNN with the parameter
    assignment capabilities of WriteParameter.
    """
    def __init__(self, graph_node_features:int=512, in_feats:int=None, in_feat_name:Union[str,List[str]]=["atomic_number", "ring_encoding", "partial_charge"], in_feat_dims:Dict[str,int]={}, gnn_width:int=None, gnn_attentional_layers:int=3, gnn_convolutions:int=3, gnn_attention_heads:int=8, gnn_dropout_attention:float=0., gnn_dropout_initial:float=0., gnn_dropout_conv:float=0., gnn_dropout_final:float=0., parameter_dropout:float=0., bond_transformer_depth=2, bond_n_heads=8, bond_transformer_width=512, bond_symmetriser_depth=2, bond_symmetriser_width=256, angle_transformer_depth=2, angle_n_heads=8, angle_transformer_width=512, angle_symmetriser_depth=2, angle_symmetriser_width=256, proper_transformer_depth=2, proper_n_heads=8, proper_transformer_width=512, proper_symmetriser_depth=2, proper_symmetriser_width=256, improper_transformer_depth=2, improper_n_heads=8, improper_transformer_width=512, improper_symmetriser_depth=2, improper_symmetriser_width=256, n_periodicity_proper=6, n_periodicity_improper=3, gated_torsion:bool=False, wrong_symmetry=False, positional_encoding=True, layer_norm=True, self_interaction=True, stat_dict:dict=get_default_statistics()):
        super().__init__()

        # Initialize GrappaGNN
        self.gnn = GrappaGNN(
            out_feats=graph_node_features,
            in_feats=in_feats,
            node_feats=gnn_width,
            n_conv=gnn_convolutions,
            n_att=gnn_attentional_layers,
            n_heads=gnn_attention_heads,
            in_feat_name=in_feat_name,
            in_feat_dims=in_feat_dims,
            conv_dropout=gnn_dropout_conv,
            attention_dropout=gnn_dropout_attention,
            final_dropout=gnn_dropout_final,
            initial_dropout=gnn_dropout_initial,
            self_interaction=self_interaction,
            layer_norm=layer_norm
        )

        # Initialize WriteParameter
        self.parameter_writer = WriteParameters(
            graph_node_features=graph_node_features,
            parameter_dropout=parameter_dropout,  # Assuming no dropout for parameter writer
            layer_norm=layer_norm,      # Assuming layer normalization for parameter writer
            positional_encoding=positional_encoding,  # Assuming positional encoding for parameter writer
            bond_transformer_depth=bond_transformer_depth, 
            bond_n_heads=bond_n_heads, 
            bond_transformer_width=bond_transformer_width, 
            bond_symmetriser_depth=bond_symmetriser_depth, 
            bond_symmetriser_width=bond_symmetriser_width,
            angle_transformer_depth=angle_transformer_depth,
            angle_n_heads=angle_n_heads,
            angle_transformer_width=angle_transformer_width,
            angle_symmetriser_depth=angle_symmetriser_depth,
            angle_symmetriser_width=angle_symmetriser_width,
            proper_transformer_depth=proper_transformer_depth,
            proper_n_heads=proper_n_heads,
            proper_transformer_width=proper_transformer_width,
            proper_symmetriser_depth=proper_symmetriser_depth,
            proper_symmetriser_width=proper_symmetriser_width,
            improper_transformer_depth=improper_transformer_depth,
            improper_n_heads=improper_n_heads,
            improper_transformer_width=improper_transformer_width,
            improper_symmetriser_depth=improper_symmetriser_depth,
            improper_symmetriser_width=improper_symmetriser_width,
            n_periodicity_proper=n_periodicity_proper,
            n_periodicity_improper=n_periodicity_improper,
            wrong_symmetry=wrong_symmetry,
            stat_dict=stat_dict,
            gated_torsion=gated_torsion
        )

    def forward(self, g):
        """
        Forward pass of the GrappaModel. Applies GrappaGNN to extract features and then WriteParameters to assign molecular parameters.

        Parameters:
            g (dgl.DGLGraph): Input graph.

        Returns:
            dgl.DGLGraph: The graph with assigned molecular parameters.
        """
        g = self.gnn(g)
        g = self.parameter_writer(g)
        return g
