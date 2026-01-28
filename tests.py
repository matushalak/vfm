import os
import torch
from numpy import array, nanmean, nanmax, nanmin
from molecule_data import QM9Dataset, show_2d, make_molecule, get_smiles
from torch_geometric.loader import DataLoader as GraphDataLoader
from my_utils import batch_to_dense, PlaceHolder, load_checkpoint
from matplotlib import pyplot as plt
from tqdm import tqdm
from vfm import sample
from vanilla_transformer import GraphTransformer
from config import get_config_net

ROOT = os.path.expanduser("data/pyg_molecules")

def test_data_loading(drop_H = False):
    '''
    The QM9 chemical data set of small molecules.

    In this dataset, nodes represent atoms and edges represent chemical bonds.
    There are 4 possible atom types if hydrogen is excluded (C, N, O, F) 
    and 4 bond types (single, double, triple, aromatic).

    Node features X, represent one-hot encoding of atomic number (atom type)
        cols 0-3, {'C': 0, 'N': 1, 'O': 2, 'F': 3}
    Edge features E, represent one-hot encoding of bond type
        cols 0-4 (single, double, triple, aromatic, no_bond)
    Coordinates C, represent X,Y,Z coordinates of the individual atoms
    '''
    # Load Dataset
    # TODO: custom train / val / test split of dataset
    qm9 = QM9Dataset(root=os.path.join(ROOT, "QM9"), split='test',
                     small_data=False, keep_pos=True, drop_H=drop_H)
    # Data Loader
    qm9_loader = GraphDataLoader(qm9, batch_size=32, shuffle=False)
    # study_distances(qm9_loader)

    # Get smiles representation of molecule
    qm9smiles = get_smiles(qm9_loader)
    # Iter through batch
    qm9_batch = next(iter(qm9_loader))
    # Preprocess batch to dense format
    batch_ready, node_mask = batch_to_dense(qm9_batch)
    Graph = PlaceHolder(batch_ready.X, batch_ready.E, batch_ready.y)
    g = Graph.mask(node_mask)
    biggest_mol = 10#torch.where(torch.all(torch.any(batch_ready.X == 1,dim = -1), dim = 1))[0][0]
    rdMol = make_molecule(batch_ready.X[biggest_mol], batch_ready.E[biggest_mol], node_mask[biggest_mol,:].sum(),
                          drop_H=drop_H)
    show_2d(rdMol, show=True)
    pass


def study_distances(loader):
    dists_max = []
    dists_min = []
    for bbb in tqdm(loader):
        assert hasattr(bbb, 'pos')
        g, node_mask = batch_to_dense(bbb)
        c = g.C
        dist = torch.sqrt(((c[:, :, None, :] - c[:, None, :, :])**2).sum(-1))
        dist_mask = node_mask[..., None] & node_mask[:, None, :]
        dist_mask &= ~torch.eye(node_mask.shape[1], device=dist.device, dtype=torch.bool)[None, :, :]
        dist[~dist_mask] = torch.nan
        dists_max.append(nanmax(dist))
        dists_min.append(nanmin(dist))

    print(f'DistMax: mean {nanmean(array(dists_max))}, max {nanmax(array(dists_max))}')
    print(f'DistMin: mean {nanmean(array(dists_min))}, min {nanmin(array(dists_min))}')

def test_checkpoint():
    path = 'runs/run_{epochs:22,bs:512,lr:0.001,drop_H:True,keep_pos:False,num_layers:6,n_molsizes:2,mol_per_molsize:50,small_model:True,small_data:True}/checkpoints/best_fcd.pt'
    model = get_config_net(num_layers=6, num_node_feats=4, num_edge_feats=5)
    ckpt = load_checkpoint(path = path, model = model) 
    breakpoint()

if __name__ == '__main__':
    test_checkpoint()
    test_data_loading(drop_H=True)