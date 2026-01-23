import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader as GraphDataLoader
from tqdm import tqdm

from architecture import E3GraphTransformer
from vanilla_transformer import GraphTransformer
from vfm import train, sample
from molecule_data import QM9Dataset, show_2d, make_molecule, get_stats, get_smiles, eval_molecules
from my_utils import batch_to_dense, PlaceHolder, get_run_name
from config import get_config_net

print('MPS is available:', torch.backends.mps.is_available())
DEVICE = 'mps' if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device(DEVICE)
print(f'Device = {DEVICE}')
ROOT = os.path.expanduser("data/pyg_molecules")
LOGDIR = os.path.expanduser('runs')

def main(args):
    # Load Dataset
    # Train / val / test split of dataset
    qm9 = QM9Dataset(root=os.path.join(ROOT, "QM9"),
                     drop_H=args.drop_H, keep_pos=args.keep_pos, 
                     split='train', small_data=args.small_data)
    # Data Loader
    qm9_loader = GraphDataLoader(qm9, batch_size=args.bs, shuffle=True)
    # initialize model
    GT = get_config_net(num_layers=args.num_layers, 
                        num_node_feats=len(qm9.atom_decoder),
                        num_edge_feats=len(qm9.bond_decoder),
                        small_model=args.small_model)
    # train model & log
    log_path = os.path.join(LOGDIR, get_run_name(args))
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)
    writer.add_text('hparams', str(vars(args)))
    train(net=GT,
          G = qm9_loader,
          incl_positions=args.keep_pos,
          TRAIN=True,
          epochs=args.epochs,
          lr = args.lr,
          writer=writer
          )
    
    molecule_stats = get_stats(drop_H=args.drop_H)
    # sample new molecules with different number of atoms 100 times
    all_gen_molecules = []
    print('Generating molecules!')
    for _ in tqdm(range(100)):
        natoms = torch.multinomial(molecule_stats['n'], num_samples=1)
        atom_feats, bond_adj = sample(n_atoms=natoms,
                                      n_samples=args.mol_per_molsize,
                                      dt = 1e-2, 
                                      net = GT,
                                      dims = dict(x = len(qm9.atom_decoder),
                                                  e = len(qm9.bond_decoder)),
                                      incl_positions=args.keep_pos)
        # make RdKit molecule
        molecules = [make_molecule(x, e, natoms) 
                    for x, e in zip(atom_feats, bond_adj)]
        all_gen_molecules += molecules
    
    # evaluate generated molecules
    train_smiles = get_smiles(qm9_loader)
    qm9 = QM9Dataset(root=os.path.join(ROOT, "QM9"),
                     drop_H=args.drop_H, keep_pos=args.keep_pos, 
                     split='test', small_data=args.small_data)
    qm9_loader = GraphDataLoader(qm9)
    test_smiles = get_smiles(qm9_loader)

    valid_molecules, results = eval_molecules(mols = all_gen_molecules, smiles=(train_smiles, test_smiles))
    # log results
    for k, v in results.items():
        try:
            writer.add_scalar(f'eval/{k}', float(v), 0)
        except Exception:
            pass
    writer.flush()
    writer.close()
    # plot valid molecules
    for mol in valid_molecules:
        show_2d(mol, save=True, SAVEDIR=log_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # General parameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--bs', type=int, default=32,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--drop_H', type=bool, default=True,
                        help='train and sample molecules without hydrogens')
    parser.add_argument('--keep_pos', type=bool, default=False,
                        help='keep 3d atom positions for training and plotting')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='number of graph transformer blocks')
    parser.add_argument('--mol_per_molsize', type=int, default=100, # 100 x 100 = 10k
                        help='number of molecules generated per each of 100 molecule sizes')
    parser.add_argument('--small_model', action='store_true', default=False,
                        help='graph transformer with small latent dimensions')
    parser.add_argument('--small_data', action='store_true', default=False,
                        help='only use 1% of the dataset (only 1000 molecules)')
    parsed_args = parser.parse_args()
    
    # run
    main(parsed_args)