from grappa.models.graph_attention import GrappaGNN
from grappa.models.interaction_parameters import WriteParameters
from typing import Union, List, Tuple, Dict
import torch
from grappa.utils.graph_utils import get_default_statistics

class GrappaModel(torch.nn.Module):
    def __init__(self,
                 graph_node_features:int=256,
                 in_feats:int=None,
                 in_feat_name:Union[str,List[str]]=["atomic_number", "ring_encoding", "partial_charge", "degree"],
                 in_feat_dims:Dict[str,int]={},
                 gnn_width:int=512,
                 gnn_attentional_layers:int=4,
                 gnn_convolutions:int=0,
                 gnn_attention_heads:int=16,
                 gnn_dropout_attention:float=0.3,
                 gnn_dropout_initial:float=0.,
                 gnn_dropout_conv:float=0.,
                 gnn_dropout_final:float=0.1,
                 symmetric_transformer_dropout:float=0.5,
                 symmetric_transformer_depth:int=1,
                 symmetric_transformer_n_heads:int=8,
                 symmetric_transformer_width:int=512,
                 symmetriser_depth:int=4,
                 symmetriser_width:int=256,
                 n_periodicity_proper:int=3,
                 n_periodicity_improper:int=3,
                 gated_torsion:bool=False,
                 positional_encoding:bool=True,
                 layer_norm:bool=True,
                 self_interaction:bool=True,
                 learnable_statistics:bool=False,
                 param_statistics:dict=get_default_statistics(),
                 torsion_cutoff:float=1.e-4,
                 harmonic_gate:bool=False,
                 only_n2_improper:bool=True,
                 stat_scaling:bool=True,
                 shifted_elu:bool=True):
        """
        Implements a grappa model which combines a Graph Neural Network (GNN) for feature extraction followed by a parameter prediction for bonds, angles, and torsions.

        The GNN part processes the input graph to extract features, which are then used by the parameter writer to assign molecular parameters such as bond lengths, angles, and torsion angles.

        ----------
        Arguments:
        ----------
            out_feats (int): Number of output features from the GNN, also used as graph node features for the parameter writer.
            in_feats (int): Number of input features for the GNN.
            in_feat_name (Union[str, List[str]]): Names of input features for the GNN, which it will extract from the input graphs during forward.
            in_feat_dims (Dict[str, int]): Dictionary mapping feature names to their dimensions for the GNN.
            gnn_width (int): Number of hidden node features in the GNN. Defaults to `out_feats`.
            gnn_attentional_layers (int): Number of attentional layers in the GNN.
            gnn_convolutions (int): Number of convolutional layers in the GNN.
            gnn_attention_heads (int): Number of attention heads in each attention block of the GNN.
            gnn_dropout_attention (float): Dropout rate in attention layers of the GNN.
            gnn_dropout_initial (float): Dropout rate after the initial linear layer of the GNN.
            gnn_dropout_conv (float): Dropout rate in convolutional layers of the GNN.
            gnn_dropout_final (float): Dropout rate in the final output layer of the GNN.
            symmetric_transformer_dropout (float): Dropout rate in the parameter writer.

            symmetric_transformer_depth (int, optional): Depth of the transformer for each parameter writer.
            symmetric_transformer_n_heads (int, optional): Number of attention heads for each parameter writer.
            symmetric_transformer_width (int, optional): Hidden feature dimension of the feed forward layers of the transformer for each parameter writer.
            symmetriser_depth (int, optional): Number of layers in the symmetriser for each parameter writer.
            symmetriser_width (int, optional): Feature dimension of the symmetriser for each parameter writer.
            n_periodicity_proper (int, optional): Number of periodicity terms for proper torsions. Defaults to 3.
            n_periodicity_improper (int, optional): Number of periodicity terms for improper torsions. Defaults to 3.
            gated_torsion (bool, optional): Flag indicating whether to use gated torsions. Defaults to True.
            positional_encoding (bool, optional): Flag indicating whether to use positional encoding in the parameter writer. Defaults to True.
            layer_norm (bool, optional): Flag indicating whether to use layer normalization in the parameter writer. Defaults to True.
            self_interaction (bool, optional): Flag indicating whether to use self-interaction in the GNN. Defaults to True.
            learnable_statistics (bool, optional): Flag indicating whether to use learnable statistics for parameter initialization. Defaults to False.
            param_statistics (dict, optional): Dictionary containing statistics for parameter initialization.
            torsion_cutoff (float, optional): Cutoff value for torsion force constants in kcal/mol. Defaults to 1e-4.
            harmonic_gate (bool, optional): Apply a scaled sigmoid gate (values between 0 and 2) to the output of angle and bond force constants.
        The GrappaModel is a composite model combining the feature extraction capabilities of GrappaGNN with the parameter
        assignment capabilities of WriteParameter.
        """
        super().__init__()

        bond_transformer_depth=symmetric_transformer_depth
        bond_n_heads=symmetric_transformer_n_heads
        bond_transformer_width=symmetric_transformer_width
        bond_symmetriser_depth=symmetriser_depth
        bond_symmetriser_width=symmetriser_width

        angle_transformer_depth=symmetric_transformer_depth
        angle_n_heads=symmetric_transformer_n_heads
        angle_transformer_width=symmetric_transformer_width
        angle_symmetriser_depth=symmetriser_depth
        angle_symmetriser_width=symmetriser_width

        proper_transformer_depth=symmetric_transformer_depth
        proper_n_heads=symmetric_transformer_n_heads
        proper_transformer_width=symmetric_transformer_width
        proper_symmetriser_depth=symmetriser_depth
        proper_symmetriser_width=symmetriser_width

        improper_transformer_depth=symmetric_transformer_depth
        improper_n_heads=symmetric_transformer_n_heads
        improper_transformer_width=symmetric_transformer_width
        improper_symmetriser_depth=symmetriser_depth
        improper_symmetriser_width=symmetriser_width

        wrong_symmetry=False

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
            symmetric_transformer_dropout=symmetric_transformer_dropout,
            layer_norm=layer_norm,
            positional_encoding=positional_encoding,
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
            param_statistics=param_statistics,
            gated_torsion=gated_torsion,
            learnable_statistics=learnable_statistics,
            torsion_cutoff=torsion_cutoff,
            harmonic_gate=harmonic_gate,
            only_n2_improper=only_n2_improper,
            stat_scaling=stat_scaling,
            shifted_elu=shifted_elu
        )

        # field of view relates to attention layers and convolutions; + 3 to get dihedrals and ring membership (up to 6 membered rings, for larger rings this should be higher)
        self.field_of_view = gnn_attentional_layers + gnn_convolutions + 3

    def forward(self, g):
        """
        Forward pass of the GrappaModel. Applies GrappaGNN to extract features and then WriteParameters to assign molecular parameters.

        Parameters:
            g (dgl.DGLGraph): Input graph.

        Returns:
            dgl.DGLGraph: The graph with assigned molecular parameters.
        """
        # first, check consistency of the graph:
        for lvl in ['n2', 'n3', 'n4', 'n4_improper']:
            if lvl in g.ntypes:
                if len(g.nodes[lvl].data['idxs']) > 0:
                    max_idx = torch.max(g.nodes[lvl].data['idxs'])
                else:
                    max_idx = 0
                assert max_idx <= g.num_nodes('n1'), f'Encountered idxs up to {max_idx} at the level g.nodes[{lvl}].data["idxs"], but there are only {g.num_nodes("n1")} atom-level-nodes in the graph'

        g = self.gnn(g)
        g = self.parameter_writer(g)
        return g
