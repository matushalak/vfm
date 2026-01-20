
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from itertools import product
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from IPython.display import display
import py3Dmol
import matplotlib.pyplot as plt

# Pad first category as No Bond
# After sampling, when inspect

class QM9Dataset(QM9):
    '''
    Without hydrogens!
    '''
    def __init__(self, root, 
                 drop_H = True, keep_pos = False, 
                 split = 'train',
                 small_data = False,
                 transform = None, pre_transform = None, pre_filter = None, force_reload = False):
        super().__init__(root, transform, pre_transform, pre_filter, force_reload)
        self.keep_pos = keep_pos
        self.drop_H = drop_H
        if drop_H:
            self.atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3}
            self.atom_decoder = ['C', 'N', 'O', 'F']
            self.atom_slice = slice(1, 1+len(self.atom_decoder))
        else:
            self.atom_encoder = {'H':0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
            self.atom_decoder = ['H', 'C', 'N', 'O', 'F']
            self.atom_slice = slice(0, len(self.atom_decoder))
        self.bond_decoder = ['None', 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        
        # Slice only the appropriate train / val / test sets
        assert split in ('train', 'val', 'test', 'full')
        n = super().__len__()
        g = torch.Generator()
        g.manual_seed(42)
        perm = torch.randperm(n, generator=g)
        # Split numbers taken from VFM paper 
        # https://openreview.net/forum?id=UahrHR5HQh&noteId=BoBuVw1Bmx
        split_slice = {'train': slice(0, 100000),
                       'val': slice(100000, 120000),
                       'test': slice(120000, n),
                       'full': slice(0, n)}
        idx = perm[split_slice[split]]
        if small_data:
            idx = idx[:1000]
        self.idx_set = idx.tolist()
        
    def __len__(self):
        return len(self.idx_set)
    
    def __getitem__(self, idx):
        graph = super().__getitem__(self.idx_set[idx])
        # Dropping hydrogen setting
        if self.drop_H:
            # non-hydrogen one-hot (assumes x columns are [H, C, N, O, F])
            non_h = graph.x[:, self.atom_slice]
            keep = ~non_h.eq(0).all(dim=1)   # True for non-H nodes

            # build mapping: old node id -> new node id (0..n_kept-1)
            n = graph.num_nodes
            mapping = torch.full((n,), -1, dtype=torch.long)
            mapping[keep] = torch.arange(int(keep.sum()), dtype=torch.long)

            # filter node features/positions
            graph.x = non_h[keep]
            graph.pos = graph.pos[keep]

            # filter edges: keep edges where both endpoints are kept
            ei = graph.edge_index
            src, dst = ei[0], ei[1]
            ekeep = keep[src] & keep[dst]

            ei = ei[:, ekeep]
            # Remap endpoints
            ei = mapping[ei]   # vectorized remap, now in [0..n_kept-1]
            graph.edge_index = ei
            graph.edge_attr = graph.edge_attr[ekeep]
        
        else:
            graph.x = graph.x[:, self.atom_slice]

        # Keeping coordinates setting
        if not self.keep_pos:
            graph.pos = None
        return graph
    
def get_stats(drop_H:bool = True)->dict:
    '''
    QM9 Stats taken from DiGress paper codebase https://github.com/cvignac/DiGress
    '''
    if drop_H: # without hydrogen
        # fraction of molecules with certain n nodes
        n_nodes = torch.tensor([0,
                                2.2930e-05, 3.8217e-05, 6.8791e-05, 
                                2.3695e-04, 9.7072e-04, 0.0046472, 
                                0.023985, 0.13666, 0.83337])
        # fraction of real nodes of given type
        node_types = torch.tensor([0.7230, 0.1151, 0.1593, 0.0026])
        # fraction of real edges of given type
        edge_types = torch.tensor([0.7261, 0.2384, 0.0274, 0.0081, 0.0])
    else:
        n_nodes = torch.tensor([0, 0, 0, 1.5287e-05, 3.0574e-05, 3.8217e-05,
                                9.1721e-05, 1.5287e-04, 4.9682e-04, 1.3147e-03, 3.6918e-03, 8.0486e-03,
                                1.6732e-02, 3.0780e-02, 5.1654e-02, 7.8085e-02, 1.0566e-01, 1.2970e-01,
                                1.3332e-01, 1.3870e-01, 9.4802e-02, 1.0063e-01, 3.3845e-02, 4.8628e-02,
                                5.4421e-03, 1.4698e-02, 4.5096e-04, 2.7211e-03, 0.0000e+00, 2.6752e-04])

        node_types = torch.tensor([0.5122, 0.3526, 0.0562, 0.0777, 0.0013])
        edge_types = torch.tensor([0.88162,  0.11062,  5.9875e-03,  1.7758e-03, 0])

    dist = dict(n = n_nodes, 
                atom_types = node_types,
                bond_types = edge_types)
    return dist

    

def make_molecule(x, e, size):
    ''' 
    Make RdKit molecule from Matrix representation without hydrogens
    Should work for QM9 and ZINC
    '''
    x = x.squeeze()
    e = e.squeeze()
    dict = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'Br': 4, 'Cl': 5, 'I': 6, 'P': 7, 'S': 8}
    atom_dict = {v: k for k, v in dict.items()}

    molecule = Chem.RWMol()
    # Add atoms based on feature matrix
    for i in range(size):
        atom = torch.argmax(x[i,:], -1).item()
        atom = Chem.Atom(atom_dict[atom])
        molecule.AddAtom(atom)

    bonds = torch.argmax(e, dim=-1)
    # Add bonds based on soft adjacency matrix
    for i, j in product(range(size), range(size)):
        if i < j:
            bond = bonds[i, j]
            if bond == 4:
                molecule.AddBond(i, j, Chem.BondType.AROMATIC)
            if bond == 3:
                molecule.AddBond(i, j, Chem.BondType.TRIPLE)
            elif bond == 2:
                molecule.AddBond(i, j, Chem.BondType.DOUBLE)
            elif bond == 1:
                molecule.AddBond(i, j, Chem.BondType.SINGLE)
    return molecule


def show_2d(mol, size=(300, 300)):
    img = Draw.MolToImage(mol, size=size)
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.show()

# TODO: change to work without display and just produce images
def show_3d(mol, style="stick"):
    mb = Chem.MolToMolBlock(mol)
    view = py3Dmol.view(width=420, height=320)
    view.addModel(mb, "mol")
    view.setStyle({style: {}})
    view.zoomTo()
    return view.show()