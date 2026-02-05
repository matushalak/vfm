from vfm import sample
from architecture import E3GraphTransformer
from vanilla_transformer import GraphTransformer
from config import get_config_net
from my_utils import load_checkpoint, suppress_console_output
from molecule_data import make_molecule, show_3d, get_stats, eval_molecules

import torch
from tqdm import tqdm

def main(path:str, nmols:int, keep_pos:bool = True):
    DEVICE = 'mps' if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device(DEVICE)
    print('Loading model!')
    model = get_config_net(num_layers=6, 
                           num_node_feats=4, num_edge_feats=5, num_coord_feats=3,
                           small_model=False, transformer=E3GraphTransformer)
    model.to(DEVICE)
    ckpt = load_checkpoint(path, model, map_location=DEVICE)
    molecule_stats = get_stats()

    # Generate molecules using pretrained model
    print('Generating molecules!')
    natoms = torch.multinomial(molecule_stats['n'], num_samples=1)
    out = sample(n_atoms=natoms,
                n_samples=nmols,
                dt = 1e-2, 
                net = ckpt,
                dims = dict(x = 4,
                            e = 5,
                            c = 3),
                incl_positions=True)
    
    if keep_pos:
        atom_feats, bond_adj, coords = out
        all_pred_coords = [c for c in coords]
    else:
        atom_feats, bond_adj = out
        all_pred_coords = None
        
    # with suppress_console_output():
    # make RdKit molecule
    molecules = [make_molecule(x, e, natoms) 
                for x, e in zip(atom_feats, bond_adj)]

    # evaluate generated molecules
    valid_molecules, valid_coords = eval_molecules(mols = molecules, 
                                                   valid_only=True,
                                                   pred_coords = all_pred_coords)
        
    # Plot
    print('Visualizing molecules!')
    for mol, coord in zip(valid_molecules, valid_coords):
        show_3d(mol, coord, show=True, s = 300, axis_off=False)

    
    


if __name__ == '__main__':
    main(path='runs/run_{epochs:1000,bs:1024,lr:0.001,drop_H:True,keep_pos:True,num_layers:6,n_molsizes:100,mol_per_molsize:100,small_model:False,small_data:False}/checkpoints/best_fcd.pt',
         nmols=100)