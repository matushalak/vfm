import os
import torch
from torch_geometric.loader import DataLoader as GraphDataLoader

from architecture import E3GraphTransformer
from vanilla_transformer import GraphTransformer
from vfm import train, sample
from molecule_data import QM9Dataset, show_2d, make_molecule, get_stats
from my_utils import batch_to_dense, PlaceHolder
from config import get_config_net

print('MPS is available:', torch.backends.mps.is_available())
DEVICE = 'mps' if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(DEVICE)
ROOT = os.path.expanduser("data/pyg_molecules")

def main(split:str = 'train'):
    # Load Dataset
    # Train / val / test split of dataset
    qm9 = QM9Dataset(root=os.path.join(ROOT, "QM9"),
                     drop_H=True, keep_pos=False, 
                     split=split, small_data=True)
    # Data Loader
    qm9_loader = GraphDataLoader(qm9, batch_size=32, shuffle=True)
    # initialize model
    GT = get_config_net(num_layers=6, 
                        num_node_feats=len(qm9.atom_decoder),
                        num_edge_feats=len(qm9.bond_decoder),
                        small_model=True)
    # train model
    train(net=GT,
          G = qm9_loader,
          incl_positions=False,
          TRAIN=True,
          epochs=20,
          lr = 3e-3)
    
    molecule_stats = get_stats(drop_H=True)
    # sample new molecules
    natoms = torch.multinomial(molecule_stats['n'], num_samples=1)
    atom_feats, bond_adj = sample(n_atoms=natoms,
                                  n_samples=5,
                                  dt = 1e-2, 
                                  net = GT,
                                  dims = dict(x = len(qm9.atom_decoder),
                                              e = len(qm9.bond_decoder)),
                                  incl_positions=False)
    molecules = [make_molecule(x, e, natoms) 
                 for x, e in zip(atom_feats, bond_adj)]

    for mol in molecules:
        show_2d(mol)

if __name__ == '__main__':
    main()