import os
import torch
from molecule_data import QM9Dataset, show_2d, make_molecule
from torch_geometric.loader import DataLoader as GraphDataLoader
from my_utils import batch_to_dense, PlaceHolder

ROOT = os.path.expanduser("data/pyg_molecules")

def test_data_loading():
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
    qm9 = QM9Dataset(root=os.path.join(ROOT, "QM9"), split='train')
    # Data Loader
    qm9_loader = GraphDataLoader(qm9, batch_size=32, shuffle=False)
    # Iter through batch
    qm9_batch = next(iter(qm9_loader))
    # Preprocess batch to dense format
    batch_ready, node_mask = batch_to_dense(qm9_batch)
    Graph = PlaceHolder(batch_ready.X, batch_ready.E, batch_ready.y)
    g = Graph.mask(node_mask)
    biggest_mol = torch.where(torch.all(torch.any(batch_ready.X == 1,dim = -1), dim = 1))[0][0]
    rdMol = make_molecule(batch_ready.X[biggest_mol], batch_ready.E[biggest_mol], node_mask[biggest_mol,:].sum())
    show_2d(rdMol)
    pass




if __name__ == '__main__':
    test_data_loading()