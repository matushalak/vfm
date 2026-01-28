import torch.nn as nn
from vanilla_transformer import GraphTransformer
from architecture import E3GraphTransformer

def get_config_net(num_layers, num_node_feats, num_edge_feats, num_coord_feats = 3,
                   small_model:bool = True, transformer = GraphTransformer
                   )->GraphTransformer|E3GraphTransformer:
    if small_model:
        hidden_dims = {'dx': 16, 'de': 8, 'dy': 8, 'n_head': 2, 'dim_ffX': 16, 'dim_ffE': 8, 'dim_ffy': 8}
        hidden_mlp_dims = {'X': 32, 'E': 16, 'y': 16}
    else:
        hidden_dims = {'dx': 128, 'de': 64, 'dy': 128, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 64, 'dim_ffy': 256}
        hidden_mlp_dims = {'X': 256, 'E': 128, 'y': 128}

    inoutdims = {'X': num_node_feats, 'E': num_edge_feats, 'y': 1}
    
    if transformer is E3GraphTransformer:
        hidden_dims['dc'] = num_coord_feats
        # RBF expansion of pairwise distances
        hidden_dims['dim_rbf'] = 16 if small_model else 32 
        hidden_dims['dim_ffCD'] = 32 if small_model else 64
        inoutdims['C'] = num_coord_feats

    model = transformer(
        input_dims=inoutdims,
        hidden_dims=hidden_dims,
        hidden_mlp_dims=hidden_mlp_dims,
        output_dims=inoutdims,
        n_layers=num_layers,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU()
        )

    return model