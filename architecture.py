import math
import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from my_utils import PlaceHolder, assert_correctly_masked

#### PNA ####
class Xtoy(nn.Module):
    ''' 
    Principal neighborhood aggregation over node features
    '''
    def __init__(self, dx, dy):
        """ Map node features to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        """ X: bs, n, dx. """
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out

class Etoy(nn.Module):
    ''' 
    Principal neighborhood aggregation over edge features
    '''
    def __init__(self, d, dy):
        """ Map edge features to global features. """
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        """ E: bs, n, n, de
            Features relative to the diagonal of E could potentially be added.
        """
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out

class Cdtoy(nn.Module):
    ''' 
    Principal neighborhood aggregation over node coordinate pairwise distances
        Permutation invariant
        Pairwise distances are E3 invariant 
            (dcd = 1 or after embedding in RBF can be high dim dcd = K >> 1)
    '''
    def __init__(self, dcd, dy):
        """ Map node coordinates to global features """
        super().__init__()
        self.lin = nn.Linear(4 * dcd, dy)

    def forward(self, Cd):
        """ Cd: bs, n, n, dcd """
        m = Cd.mean(dim=(1, 2))
        mi = Cd.min(dim=2)[0].min(dim=1)[0]
        ma = Cd.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(Cd, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        out = self.lin(z)
        return out

def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


class dRBF(nn.Module):
    '''
    Radial Basis Function Embedding for E3-invariant pairwise distances
    '''
    def __init__(self, n_bases:int, distance_range:tuple = (0, 11)):
        super().__init__()
        self.rmin, self.rmax = distance_range
        self.centers = torch.linspace(*distance_range, steps=n_bases)
        delta = (distance_range[1]-distance_range[0]) / (n_bases-1)
        self.gamma = 1/(2*(delta**2) + 1e-12)

    def forward(self, dist:torch.Tensor)->torch.Tensor:
        ''''
        computes RBF embeddings for each of the E3-invariant pairwise distances
            u_i = exp(-gamma * (d - µ_i)**2)
        
        dist: (bs, n, n)
        µ: (K,)

        Returns
        u: (bs, n, n, K)
        '''
        self.centers = self.centers.to(dtype=dist.dtype, device=dist.device)
        diff2 = (dist[..., None] - self.centers[None, None, None, :]) ** 2 # (bs, n, n, K)
        return torch.exp(-self.gamma * diff2)


class XECyTransformerLayer(nn.Module):
    """ 
    Transformer that updates node, edge, coordinate and global features. 
    This layer is permutation equivariant wrt. node ordering, and E3 equivariant wrt node 3d coordinates
        d_x: node features
        d_e: edge features
        d_c: node coordinates
        dz : global features
        n_head: the number of heads in the multi_head_attention

        dim_rbf: dimension of RBF embedding of E3-invariant pairwise coordinate distances
        dim_ffCD: dimension of MLP RBF embedding expansion BEFORE self-attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dc: int, dy: int, n_head: int, 
                 dim_ffX: int = 2048, dim_ffE: int = 128, dim_ffy: int = 2048, 
                 dim_rbf: int = 16, dim_ffCD: int = 32,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-5, 
                 device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        # Self attention block
        self.self_attn = E3NodeEdgeCoordBlock(dx, de, dc, dim_rbf, dy, n_head, **kw)

        # Linear projection layers, layernorms & dropouts for node features
        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        # Linear projection layers, layernorms & dropouts for edge features
        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        # RBF and Linear projection layers, layernorms & dropouts for E3-invariant distance features
        self.rbf = dRBF(n_bases=dim_rbf)
        self.linCD1 = Linear(dim_rbf, dim_ffCD, **kw)
        self.linCD2 = Linear(dim_ffCD, dim_rbf, **kw)
        self.normCD = LayerNorm(dim_rbf, eps=layer_norm_eps, **kw)
        self.dropoutCD1 = Dropout(dropout)
        self.dropoutCD2 = Dropout(dropout)

        # Linear projection layers, layernorms & dropouts for global graph features
        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, X: Tensor, E: Tensor, C: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            C: (bs, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """
        # mask out invalid pairwise distances / relative vectors (to padded nodes)
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        pair_mask &= ~torch.eye(C.shape[1], device=C.device, dtype=torch.bool)[None]
        
        # relative_vects (E3 equivariant)
        rel_c = C[:, :, None, :] - C[:, None, :, :] # (b, n, 1, 3) - (b, 1, n, 3) => (b, n, n, 3)
        # pairwise dist atom distances (E3 invariant)
        CD = torch.sqrt((rel_c**2).sum(-1) + 1e-12) # (b, n, n, 3) -> (b, n, n)
        # normalize relative vectors to unit length
        rel_c = rel_c / CD.unsqueeze(-1)
        
        # mask out invalid pairs
        rel_c = rel_c*pair_mask[..., None].to(rel_c.dtype)
        CD = CD.masked_fill(~pair_mask, self.rbf.rmax) # dist to padded nodes > maximum possible distance
        
        # RBF(dist) expansion on E3 invariant
        U = self.rbf(CD) # (b, n, n) -> (b, n, n, K)
        # Feedforward MLP enrichment of E3 invariant rbf features
        ff_outputU = self.linCD2(self.activation(self.linCD1(self.dropoutCD1(U))))
        U = self.normCD(U + self.dropoutCD2(ff_outputU))

        # Pass through self-attention block
        newX, newE, newC, new_y = self.self_attn(X=X, E=E, C=C, y=y, 
                                                 U=U, rel=rel_c,
                                                 node_mask=node_mask)
        
        C = newC * node_mask[..., None].to(newC.dtype) + C * (~node_mask[..., None]).to(newC.dtype)

        # Post self-attention dropout, layernorm and residual connections
        # Dropout -> Layernorm & residual connection node features
        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)
        # Dropout -> Layernorm & residual connection edge features
        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)
        # Dropout -> Layernorm & residual connection global features
        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        # Feed-forward MLP expansion with dropout, layernorm & residual connection
        # node features
        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)
        # edge features
        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)
        # global features
        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, C, y


class E3NodeEdgeCoordBlock(nn.Module):
    """ 
    Self attention layer that also updates the representations on the edges
    and performs E3-equivariant update on the 3d node coordinates. 
    """
    def __init__(self, dx, de, dc, du, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dc = dc
        self.du = du
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # E3-equivariant coordinate update modulated by attention (who) and scalar gate (how much)
        # normalized relative vectors decide direction
        ds1 = 2*dx + de + du + dy
        # scalar gate per pair & head
        self.s_ij_mlp = nn.Sequential(Linear(ds1, 2*ds1),
                                      nn.SiLU(),
                                      Linear(2*ds1, n_head),
                                      nn.Tanh())
        # time-dependent learning rate per head eta_h
        self.eta_mlp = nn.Sequential(Linear(dy, 2*dy),
                                     nn.SiLU(),
                                     Linear(2*dy, n_head),
                                     nn.Softplus())

        # FiLM E to X 
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)
        # (concat U to E) 
        # self.e_add = Linear(de+du, dx) # removed - mode collapse
        # self.e_mul = Linear(de+du, dx) # removed - mode collapse

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y - Global Graph features (time + PNA of node, edge feats + coords)
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)
        self.c_y = Cdtoy(du, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        # self.c_out = Linear(dc, dc) # removed - mode collapse
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, C, U, rel, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param C: bs, n, 3        node coordinates
        :param U: bs, n, n, K     enriched pairwise distances
        :param rel: bs, n, n, 3   relative vectors
        :param y: bs, dz          global graph features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 0. Concatenate E and U (incorporate enriched distances as edge features)
        # didn't work, mode collapse
        # EU = torch.cat((E, U), dim = -1) # (bs, n, n, de) -> (bs, n, n, de+du)
        EU = E 

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        assert_correctly_masked(Q, x_mask)

        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df
        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(EU) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((EU.size(0), EU.size(1), EU.size(2), self.n_head, self.df))

        E2 = self.e_add(EU) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((EU.size(0), EU.size(1), EU.size(2), self.n_head, self.df))

        # Incorporate edge features (enriched with distances) to self attention scores. (FiLM E & QK.T)
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E (FiLM y & E)
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. OUTER PRODUCT attn is still (bs, n, n, n_head, df) !!!
        # preserves feature dimension!!!
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head, df !!!

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                            # (bs, 1, n, n_head, df)

        # Compute scaled DOT-PRODUCT attention for coord updates (bs, n, n, n_head)
        # need to collapse feature dimension!!!
        # transform queries and keys to (bs, n_heads, n, df)
        Qh = (self.q(X) * x_mask).reshape(bs, n, self.n_head, self.df)
        Kh = (self.k(X) * x_mask).reshape(bs, n, self.n_head, self.df)
        q = Qh.permute(0, 2, 1, 3)   # (bs, h, n, df)
        k = Kh.permute(0, 2, 1, 3)   # (bs, h, n, df)

        # scaled dot-product attention, collapses df!!!
        sdp_attn = (q @ k.transpose(-2, -1)) / (self.df**(1/2)) # (bs, n_heads, n, n)
        sdp_attn = torch.permute(sdp_attn, (0, 2, 3, 1)) # (bs, n, n, n_heads)
        sdp_attn = masked_softmax(sdp_attn, softmax_mask, dim = 2) # (bs, n, n, n_heads) <- use this as a_ij!!!

        # Compute multi-headed scalar gate s_ij
        Xi = X.unsqueeze(2).expand(-1, -1, n, -1)         # (bs,n,n,dx)
        Xj = X.unsqueeze(1).expand(-1, n, -1, -1)         # (bs,n,n,dx)
        yij = y[:, None, None, :].expand(-1, n, n, -1)    # (bs,n,n,dy)
        gate_in = torch.cat([Xi, Xj, E, U, yij], dim=-1)  # (bs,n,n, 2dx+de+du+dy)
        s_ij = self.s_ij_mlp(gate_in) # (bs, n, n, n_head)
        # Compute per head learning rate
        eta_h = 0.1* self.eta_mlp(y) # (bs, n_head)
        # Perform update ∆c_i = ∑_h eta_h ∑_j a_ij s_ij r_ij
        # a * s * r_hat: (bs, n, n, nhead, 1) * (bs, n, n, 1, 3) -> (bs, n, n, nhead, 3)
        # ∑_j a * s * r_hat: (bs, n, n, nhead, 3) -> (bs, n, nhead, 3)
        delta_C_head = (sdp_attn[..., None] * s_ij[..., None] * rel[:, :, :, None, :]).sum(dim=2)
        # ∑_h eta_h * ∆c_h: ∑_h (bs, 1, nhead, 1) * (bs, n, nhead, 3) -> (bs, n, 3)
        delta_C = (eta_h[:, None, :, None] * delta_C_head).sum(dim=2)
        # Obtain new coordinates new_C = C + ∆C
        newC = C + delta_C

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X (FiLM y & X)
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        assert_correctly_masked(newX, x_mask)

        # Process y based on X, E and expanded E3-invariant pairwise distances U
        y = self.y_y(y)
        # PNA + linear layer to fit dim_y
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        # c_y = self.c_y(U) # removed - mode collapse
        new_y = y + x_y + e_y #+ c_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, newC, new_y


class E3GraphTransformer(nn.Module):
    """
    Extension of DiGress Graph Transformer to E3-equivariant atom coordinate updates

    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in, act_fn_out):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_C = output_dims['C']
        self.out_dim_y = output_dims['y']

        # Lift discrete features to higher latent dimension (NOT COORDS to preserve equivariance!)
        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        # Graph Transformer blocks
        self.tf_layers = nn.ModuleList([XECyTransformerLayer(dx=hidden_dims['dx'],
                                                             de=hidden_dims['de'],
                                                             dc=hidden_dims['dc'],
                                                             dy=hidden_dims['dy'],
                                                             n_head=hidden_dims['n_head'],
                                                             dim_ffX=hidden_dims['dim_ffX'],
                                                             dim_ffE=hidden_dims['dim_ffE'],
                                                             dim_rbf = hidden_dims['dim_rbf'],
                                                             dim_ffCD = hidden_dims['dim_ffCD'],
                                                             dim_ffy=hidden_dims['dim_ffy'])
                                        for i in range(n_layers)])

        # Map discrete features back to data dimension (COORDS stay 3D throughout!)
        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def forward(self, X, E, C, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        # Copy network input
        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        # Map discrete features to model dimensions
        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        after_in = PlaceHolder(X=self.mlp_in_X(X), E=new_E, C = C, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, C, y = after_in.X, after_in.E, after_in.C, after_in.y

        # Run chain of transformer blocks
        for layer in self.tf_layers:
            X, E, C, y = layer(X, E, C, y, node_mask)

        # Map to discrete features to output dimensions
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        # Residual connection around the entire latent processing stack (around all hidden layers)
        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out
        # Re-symmetrize inferred adjacency
        E = 1/2 * (E + torch.transpose(E, 1, 2))

        return PlaceHolder(X=X, E=E, C=C, y=y).mask(node_mask)
