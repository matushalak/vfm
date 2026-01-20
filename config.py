import torch.nn as nn
from vanilla_transformer import GraphTransformer

def get_config_net(num_layers, num_node_feats, num_edge_feats, 
                   small_model:bool = True
               )->GraphTransformer:
    if small_model:
        hidden_dims = {'dx': 16, 'de': 8, 'dy': 8, 'n_head': 2, 'dim_ffX': 16, 'dim_ffE': 8, 'dim_ffy': 8}
        hidden_mlp_dims = {'X': 32, 'E': 16, 'y': 16}
    else:
        hidden_dims = {'dx': 128, 'de': 64, 'dy': 128, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256}
        hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 128}

    model = GraphTransformer(
        input_dims={'X': num_node_feats, 'E': num_edge_feats, 'y': 1},
        hidden_dims=hidden_dims,
        hidden_mlp_dims=hidden_mlp_dims,
        output_dims={'X': num_node_feats, 'E': num_edge_feats, 'y': 1},
        n_layers=num_layers,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU()
        )

    return model