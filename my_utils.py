import os
import contextlib

import torch
import torch.utils as torch_utils
from torch_geometric.utils import to_dense_batch, remove_self_loops, to_dense_adj

# adapted from DiGress paper https://github.com/cvignac/DiGress
def batch_to_dense(BATCH):
    ''''
    Receives single batch from PyG dataloader
    Use in training loop before passing input into network
    '''
    x, edge_index, edge_attr, batch = BATCH.x, BATCH.edge_index, BATCH.edge_attr, BATCH.batch
    # True: node, False: no node
    X, node_mask = to_dense_batch(x=x, batch=batch) # (total_nodes, dx), (total_nodes) -> (bs, nmax, dx), (bs, nmax)
    if hasattr(BATCH, 'pos') and getattr(BATCH, 'pos') is not None:
        C, _ = to_dense_batch(x = BATCH.pos, batch=batch)
    else:
        C = None
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None, C = C), node_mask

def encode_no_edge(E):
    '''
    E is b, nmax, nmax, nbondtypes
    Add no edge as first dimension (like in DiGress and VFM)
    '''
    assert len(E.shape) == 4 and E.shape[-1] == 4
    if E.shape[-1] == 0:
        return E
    where_no_edge = torch.where(torch.sum(E, dim=3) == 0, 1, 0).unsqueeze(-1)
    E = torch.cat((where_no_edge, E), dim = -1)
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E

class PlaceHolder:
    def __init__(self, X, E, y, C = None):
        '''
        Node features (atom class)
        Edge features (bond class)
        y graph-level features (spectral, structural, time)
        C node coordinates (only for continuous / joint molecular graph generation)
        '''
        self.X = X
        self.E = E
        self.y = y
        self.C = C

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        if self.C is not None:
            self.C = self.C.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
            if self.C is not None:
                self.C[node_mask == 0] = None
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            if self.C is not None:
                self.C = self.C * x_mask
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask.long())).abs().max().item() < 1e-4, \
        'Variables not masked properly.'

def get_run_name(args)->str:
    name = str(vars(args))
    name = name.replace(' ', '').replace("'", "")
    return 'run_'+ name

### Structural features, taken from VFM paper https://openreview.net/forum?id=UahrHR5HQh&noteId=BoBuVw1Bmx
# but based on DiGress https://github.com/cvignac/DiGress
def add_feats(graph:PlaceHolder, t, mask):
    ''' 
    Add structural features to node features
    '''
    x_t, E_t, = graph.X, graph.E
    device = x_t.device
    # graph level time feature
    y_t = t.squeeze().unsqueeze(-1).to(device)
    # node mask
    mask = mask.reshape(-1, x_t.size(0)).to(device).bool()
    # weighed adjacency from E (prob of real edge, relies on softmaxed input)
    A = E_t[..., 1:].sum(dim=-1).float()
    A = A * mask.unsqueeze(1) * mask.unsqueeze(2)
    
    # Laplacian & spectral features graph-level (eigvals) and node-level (eigvects)
    L = compute_laplacian(A, normalize=False)
    mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
    mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
    L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

    eigvals, eigvectors = torch.linalg.eigh(L)
    eigenvalues = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
    eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)

    y_a, y_b = get_eigenvalues_features(eigenvalues, k=5)
    not_lcc_indicator, first_k_ev = get_eigenvectors_features(eigvectors, mask, y_a, k=2)

    y_t = torch.cat([y_t, y_a, y_b], dim=-1)
    x_t = torch.cat([x_t, not_lcc_indicator, first_k_ev], dim=-1)

    return x_t, y_t

def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)     # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)      # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency                        # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)            # (bs, n)
    D_norm = torch.diag_embed(diag_norm)        # (bs, n, n)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2

def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep

    Returns:
        n_connected_components = number of 0 eigenvalues
        first_k nonzero eigenvalues
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    assert (n_connected_components > 0).all(), (n_connected_components, ev)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev

def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
            the eigenvectors are n-dimensional, so each node gets one dimension of each eigenvector as positional encoding
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask                        # bs, n
    # Add random value to the mask to prevent 0 from becoming the mode
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)                                   # bs, n
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values                                    # values: bs -- indices: bs
    mask = ~ (first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)   # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)    # bs, 1, k
    indices = indices.expand(-1, n, -1)                                               # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)       # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev

#### Supress console output
@contextlib.contextmanager
def suppress_console_output(suppress_stdout=True, suppress_stderr=True):
    devnull = os.open(os.devnull, os.O_WRONLY)

    old_stdout_fd = os.dup(1) if suppress_stdout else None
    old_stderr_fd = os.dup(2) if suppress_stderr else None

    try:
        if suppress_stdout:
            os.dup2(devnull, 1)
        if suppress_stderr:
            os.dup2(devnull, 2)
        yield
    finally:
        if suppress_stdout and old_stdout_fd is not None:
            os.dup2(old_stdout_fd, 1)
            os.close(old_stdout_fd)
        if suppress_stderr and old_stderr_fd is not None:
            os.dup2(old_stderr_fd, 2)
            os.close(old_stderr_fd)
        os.close(devnull)

# Checkpointing
def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None):
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    
    ckpt = {
        "model": model.state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()

    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return model