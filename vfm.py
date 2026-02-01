import os
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from architecture import E3GraphTransformer
from vanilla_transformer import GraphTransformer
from my_utils import batch_to_dense, PlaceHolder, suppress_console_output, save_checkpoint
from molecule_data import eval_molecules, make_molecule, get_stats

DEVICE = 'mps' if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(DEVICE)

def train(net:E3GraphTransformer|GraphTransformer, G:GraphDataLoader, 
          incl_positions:bool = False,
          TRAIN:bool = True,
          epochs:int = 200, lr:float = 5e-4,
          writer=None, smiles:tuple|None = None, drop_H:bool = True):
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
        pos_objective = nn.MSELoss(reduction='mean')
    
    net.to(DEVICE)
    if TRAIN:
        net.train()
    else:
        net.eval()
    
    best_validity = 0
    best_fcd = 1e8

    for e in tqdm(range(epochs)):
        l, a, b, c = 0,0,0,0
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
                mu_theta = net(Xt, Et, Ct, T[:, None], node_mask) # forward pass
            else:
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
            # TODO: check masking of loss & coord output!!!
            if incl_positions:
                # train on invariant loss (pairwise distance matrix)
                # relative_vects (E3 equivariant)
                pred_rel_c = mu_theta.C[:, :, None, :] - mu_theta.C[:, None, :, :] # (b, n, 1, 3) - (b, 1, n, 3) => (b, n, n, 3)
                rel_c = c1[:, :, None, :] - c1[:, None, :, :] # (b, n, 1, 3) - (b, 1, n, 3) => (b, n, n, 3)
                # pairwise dist atom distances (E3 invariant)
                predCD = (pred_rel_c**2).sum(-1) # (b, n, n, 3) -> (b, n, n)
                CD = (rel_c**2).sum(-1) # (b, n, n, 3) -> (b, n, n)

                pred_cd = predCD.reshape(-1) # (bs*nmax*nmax)
                target_cd = CD.reshape(-1) # (bs*nmax*nmax)
                pos_loss = pos_objective(pred_cd[mask_e], target_cd[mask_e])
            
                # Combine objectives
                loss = atom_loss + 5*bond_loss + 0.01*pos_loss
                c += pos_loss.detach()
            
            else:
                # Combine objectives
                loss = atom_loss + 5*bond_loss
            
            a += atom_loss.detach()
            b += bond_loss.detach()
            l += loss.detach()

            if TRAIN:
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                # keep "last model"
                save_checkpoint(os.path.join(writer.log_dir, 'checkpoints', 'last.pt'),
                                model=net,
                                optimizer=optimizer,
                                scheduler=lr_scheduler,
                                epoch=e)
        
        # Tensorboard logging per epoch
        if writer is not None:
            writer.add_scalar('train/atom_loss', float(a.cpu()) / len(G), e)
            writer.add_scalar('train/bond_loss', float(b.cpu()) / len(G), e)
            writer.add_scalar('train/total_loss', float(l.cpu()) / len(G), e)
            if incl_positions:
                writer.add_scalar('train/pos_loss', float(c.cpu()) / len(G), e)
                writer.add_scalar('train/pairwise_distance_mean', pred_cd.mean().detach().cpu(), e)
                writer.add_scalar('train/pairwise_distance_var', pred_cd.var().detach().cpu(), e)
            writer.add_scalar('train/lr', float(optimizer.param_groups[0]['lr']), e)

            if e % 10 == 0 and smiles is not None:
                # Eval sample quality
                net.eval()
                # sample new molecules with different number of atoms 4 times
                all_gen_molecules = []
                all_pred_coords = []
                with suppress_console_output():
                    for _ in range(5):
                        natoms = torch.multinomial(get_stats(drop_H)['n'], num_samples=1)
                        out = sample(n_atoms=natoms,
                                    n_samples=200,
                                    dt = 1e-2, 
                                    net = net,
                                    dims = dict(x = len(G.dataset.atom_decoder),
                                                e = len(G.dataset.bond_decoder),
                                                c = 3),
                                    incl_positions=incl_positions)
                        
                        if incl_positions:
                            atom_feats, bond_adj, coords = out
                            all_pred_coords += [c for c in coords]
                        else:
                            atom_feats, bond_adj = out
                            all_pred_coords = None

                        # make RdKit molecule
                        molecules = [make_molecule(x, e, natoms) 
                                    for x, e in zip(atom_feats, bond_adj)]
                        all_gen_molecules += molecules
                    
                    # evaluate generated molecules
                    valid_molecules, results = eval_molecules(mols = all_gen_molecules, 
                                                              smiles=smiles,
                                                              pred_coords = all_pred_coords)
                
                # log results & save best models
                print(f'\nEpoch {e} gen results:\n', results)
                if results['Valid'] > best_validity:
                    save_checkpoint(os.path.join(writer.log_dir, 'checkpoints', 'best_validity.pt'),
                                    model=net,
                                    optimizer=optimizer,
                                    scheduler=lr_scheduler,
                                    epoch=e)
                    best_validity = results['Valid']
                
                if results['FCD'] < best_fcd:
                    save_checkpoint(os.path.join(writer.log_dir, 'checkpoints', 'best_fcd.pt'),
                                    model=net,
                                    optimizer=optimizer,
                                    scheduler=lr_scheduler,
                                    epoch=e)
                    best_fcd = results['FCD']

                for k, v in results.items():
                    try:
                        writer.add_scalar(f'validation/{k}', float(v), e)
                    except Exception:
                        pass
        if not incl_positions:
            print(f'\nEpoch {e}, Loss: {l / len(G)}, atom: {a / len(G)}, bond: {b / len(G)}\n')
        else:
            print(f'\nEpoch {e}, Loss: {l / len(G)}, atom: {a / len(G)}, bond: {b / len(G)}, position: {c / len(G)}\n')
        if TRAIN:
            lr_scheduler.step()

# TODO adjust for outputting and evaluating positions
def sample(n_atoms:int, 
           n_samples:int, dt:float, 
           net:E3GraphTransformer|GraphTransformer, 
           dims:dict,
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
            if incl_positions:
                mu_t = net(xt, et, ct, t_samples, node_mask)
            else:
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
        
        if incl_positions:
            return xt, et, ct
        else:
            return xt, et
    