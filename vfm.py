import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as GraphDataLoader
from tqdm import tqdm

from architecture import E3GraphTransformer
from vanilla_transformer import GraphTransformer
from my_utils import batch_to_dense, PlaceHolder

DEVICE = 'mps' if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(DEVICE)

def train(net:E3GraphTransformer, G:GraphDataLoader, 
          incl_positions:bool = False,
          TRAIN:bool = True,
          epochs:int = 200, lr:float = 5e-4):
    '''
    VFM training for Discrete / Joint Graph Generation
    t=0 pure noise, t=1 graph from training distribution
    '''
    # define optimizer
    optimizer = optim.AdamW(net.parameters(), lr = lr, weight_decay=1e-12)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    # Define loss functions
    atom_objective = nn.CrossEntropyLoss()
    edge_objective = nn.CrossEntropyLoss()
    if incl_positions:
        pos_objective = nn.MSELoss()
    
    net.to(DEVICE)
    if TRAIN:
        net.train()
    else:
        net.eval()

    for e in tqdm(range(epochs)):
        l = 0
        for g1 in G:
            optimizer.zero_grad()
            # Preprocess batch to dense format
            g1 = g1.to(DEVICE)
            g1_dense, node_mask = batch_to_dense(g1)
            g1_dense:PlaceHolder = g1_dense.mask(node_mask)
            x1, e1 = g1_dense.X, g1_dense.E
            ngraphs = x1.size(0)
            # sample time uniformly U(0,1)
            T = torch.rand(ngraphs, device=x1.device) # (bs)
            # NOTE: need to consider if we want to add graph-level feat
            # sample starting points randomly from Gaussian N(0,1)
            X0 = torch.randn_like(x1, device=x1.device) # (bs, nmax, natoms)
            E0 = torch.randn_like(e1, device=x1.device) # (bs, nmax, nmax, nbonds)
            # Linear Interpolants between noise and endpoints
            Xt = ((1-T)[:, None, None] * X0) + (T[:, None, None]*x1)
            Et = ((1-T)[:, None, None, None] * E0) + (T[:, None, None, None]*e1)
            if incl_positions:
                c1 = g1_dense.C
                C0 = torch.randn_like(c1, device=x1.device) # (bs, nmax, 3)
                Ct = ((1-T)[:, None, None] * C0) + (T[:, None, None]*c1)

            # Expectation (mu_theta) over Predicted Variational distribution q_theta over vector endpoints
            mu_theta = net(Xt, Et, T[:, None], node_mask) # forward pass
            
            # Losses: Ground truth Target vs Expectation of Variational Dist over target
            # Nodes (atoms)
            n_atoms = mu_theta.X.size(-1)
            pred_x = mu_theta.X.reshape(-1, n_atoms)  # (bs*nmax, n_atoms)
            target_x1 = x1.argmax(dim=-1).reshape(-1).long() # (bs*nmax) = ground-truth atom categories for CE
            mask_x = node_mask.reshape(-1) # (bs*nmax)
            atom_loss = atom_objective(pred_x[mask_x], target_x1[mask_x]) # compute loss on real unpadded nodes only!

            # Edges (bonds)
            n_bonds = mu_theta.E.size(-1)
            pred_e = mu_theta.E.reshape(-1, n_bonds)  # (bs*nmax*nmax, n_bonds)
            target_e1 = e1.argmax(dim=-1).reshape(-1).long() # (bs*nmax*nmax) = ground-truth bond categories for CE
            bs, nmax = node_mask.shape
            diag = torch.eye(nmax, device=node_mask.device, dtype=torch.bool)[None] # drop self-bonds
            edge_mask = (node_mask[:, :, None] & node_mask[:, None, :]) & (~diag) # only edges between real unpadded nodes!
            triu = torch.triu(torch.ones(nmax, nmax, device=node_mask.device, dtype=torch.bool), diagonal=1)[None] 
            edge_mask = edge_mask & triu # avoid double-counting undirected edges by keeping only upper triangular part (row<col)
            mask_e = edge_mask.reshape(-1)
            bond_loss = edge_objective(pred_e[mask_e], target_e1[mask_e])
            
            # 3D atomic coordinates
            if incl_positions:
                n_coords = mu_theta.C.size(-1) 
                pred_c = mu_theta.C.reshape(-1, n_coords) # (bs*nmax, n_coords=3)
                target_c1 = c1.reshape(-1, n_coords) # (bs*nmax, n_coords=3)
                pos_loss = pos_objective(pred_c[mask_x], target_c1[mask_x])
            
            # Combine objectives
            loss = atom_loss + 5*bond_loss
            l += loss / ngraphs
            if TRAIN:
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
        print(f'Epoch {e}, Loss: {l}')


def sample(n_atoms:int, 
           n_samples:int, dt:float, 
           net:E3GraphTransformer, dims:dict,
           incl_positions:bool = False):
    '''
    VFM sampling / generation for Discrete/Joint Molecular Generation
    '''
    with torch.no_grad():
        net.eval()
        T = torch.arange(0,1, dt, device=DEVICE)
        # at t=0 start with gaussian noise
        xt = torch.randn((n_samples, n_atoms, dims['x']), device=DEVICE)
        et = torch.randn((n_samples, n_atoms, n_atoms, dims['e']), device=DEVICE)
        if incl_positions:
            ct = torch.randn((n_samples, n_atoms, dims['c']), device=DEVICE)
        node_mask = torch.full((n_samples, n_atoms), fill_value=True, 
                               dtype=torch.bool, device=DEVICE)
        for t in T:
            # get expectation of variational dist over targets
            t_samples = t.expand(n_samples, 1)
            mu_t = net(xt, et, t_samples, node_mask)
            mu_t_x = torch.softmax(mu_t.X, dim=-1)
            mu_t_e = torch.softmax(mu_t.E, dim=-1)
            # Construct vector field
            vt_x = (mu_t_x - xt) / torch.clamp(1-t, min = 0.05)
            vt_e = (mu_t_e - et) / torch.clamp(1-t, min = 0.05)
            # move along vector field (integrate ODE)
            xt += (vt_x * dt)
            et += (vt_e * dt)
            if incl_positions:
                vt_c = (mu_t.C - ct) / torch.clamp(1-t, min = 0.05)
                ct += (vt_c * dt)
        
        return xt, et
    