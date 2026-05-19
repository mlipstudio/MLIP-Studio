"""
Prediction script: load a trained model and predict HOMO-LUMO gaps for new molecules.

Usage
-----
    python predict.py \
        --model_file best_model.pt \
        --input_file molecules.xyz \
        --output_file predictions.csv

The input can be a single or multi-frame xyz/extxyz file.
"""

import argparse
import torch
import numpy as np
from ase.io import read

from data import atoms_to_graph, gaussian_rbf, NUM_ELEMENTS
from model import MPNN
from torch_geometric.data import DataLoader, Data


def load_model(model_path: str, device: torch.device) -> MPNN:
    """Load a trained MPNN model from a checkpoint."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args_dict = checkpoint['args']
    target_mean = checkpoint['target_mean']
    target_std = checkpoint['target_std']

    model = MPNN(
        n_atom_features=NUM_ELEMENTS,
        n_edge_features=args_dict.get('n_gaussians', 64),
        node_dim=args_dict.get('node_dim', 128),
        hidden_dim=args_dict.get('hidden_dim', 128),
        n_mp_layers=args_dict.get('n_mp_layers', 2),
        readout_hidden=args_dict.get('readout_hidden', 64),
        target_mean=target_mean,
        target_std=target_std,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', '?')}, "
          f"val MAE {checkpoint.get('val_mae', '?'):.4f} eV)")
    return model, args_dict


@torch.no_grad()
def predict_from_file(
    model: MPNN,
    input_file: str,
    cutoff: float = 5.0,
    device: torch.device = torch.device('cpu'),
    batch_size: int = 64,
) -> list:
    """
    Predict band gaps for all structures in an xyz file.

    Returns list of dicts: [{'index': i, 'formula': ..., 'n_atoms': ..., 'predicted_bandgap_eV': ...}, ...]
    """
    atoms_list = read(input_file, index=':')
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]

    graphs = []
    valid_indices = []
    for i, atoms in enumerate(atoms_list):
        try:
            g = atoms_to_graph(atoms, cutoff=cutoff, target_key=None)
            graphs.append(g)
            valid_indices.append(i)
        except Exception as e:
            print(f"  Warning: skipping structure {i}: {e}")

    if not graphs:
        print("No valid structures to predict on.")
        return []

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    all_preds = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        all_preds.append(pred.cpu().numpy())

    all_preds = np.concatenate(all_preds)

    results = []
    for j, (idx, pred) in enumerate(zip(valid_indices, all_preds)):
        atoms = atoms_list[idx]
        results.append({
            'index': idx,
            'formula': atoms.get_chemical_formula(),
            'n_atoms': len(atoms),
            'predicted_bandgap_eV': float(pred),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Predict HOMO-LUMO gaps with trained MPNN')
    parser.add_argument('--model_file', type=str, required=True, help='Path to saved model .pt file')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input xyz/extxyz file')
    parser.add_argument('--output_file', type=str, default='predictions.csv', help='Output CSV file')
    parser.add_argument('--cutoff', type=float, default=None,
                        help='Distance cutoff (default: use value from training)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    model, train_args = load_model(args.model_file, device)
    cutoff = args.cutoff if args.cutoff is not None else train_args.get('cutoff', 5.0)

    results = predict_from_file(model, args.input_file, cutoff=cutoff, device=device)

    # Print results
    print(f"\n{'Index':>6} | {'Formula':>15} | {'N_atoms':>7} | {'Predicted Gap (eV)':>18}")
    print("-" * 55)
    for r in results:
        print(f"{r['index']:6d} | {r['formula']:>15} | {r['n_atoms']:7d} | {r['predicted_bandgap_eV']:18.4f}")

    # Save CSV
    import csv
    with open(args.output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'formula', 'n_atoms', 'predicted_bandgap_eV'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nPredictions saved to {args.output_file}")


if __name__ == '__main__':
    main()