"""
Data loading and graph construction from extxyz files.
Converts molecular structures into graphs with distance-based edge features
and atomic number-based node features.
"""

import numpy as np
import torch
from torch_geometric.data import Data, Dataset, DataLoader
from ase.io import read
from ase.data import atomic_numbers
from scipy.spatial.distance import pdist, squareform
from typing import List, Optional, Tuple


# Supported elements and their embedding indices
SUPPORTED_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
]
NUM_ELEMENTS = len(SUPPORTED_ELEMENTS)
ELEMENT_TO_IDX = {el: i for i, el in enumerate(SUPPORTED_ELEMENTS)}


def atoms_to_graph(
    atoms,
    cutoff: float = 5.0,
    target_key: Optional[str] = 'bandgap_eV',
    max_neighbors: Optional[int] = None,
) -> Data:
    """
    Convert an ASE Atoms object into a PyG Data graph.

    Node features: one-hot encoding of atomic species (rotationally invariant).
    Edge features: interatomic distances passed through a Gaussian RBF expansion
                   (rotationally invariant scalars).
    Edges: all pairs within a distance cutoff.

    Parameters
    ----------
    atoms : ase.Atoms
        The molecular structure.
    cutoff : float
        Distance cutoff in Angstrom for building edges.
    target_key : str or None
        Key in atoms.info that holds the target property.
    max_neighbors : int or None
        If set, keep at most this many nearest neighbors per atom.

    Returns
    -------
    torch_geometric.data.Data
    """
    n_atoms = len(atoms)
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()  # (N, 3)

    # --- Node features: one-hot atomic number ---
    node_features = []
    for sym in symbols:
        idx = ELEMENT_TO_IDX.get(sym, None)
        if idx is None:
            raise ValueError(
                f"Element '{sym}' not in supported list. "
                f"Add it to SUPPORTED_ELEMENTS in data.py."
            )
        one_hot = np.zeros(NUM_ELEMENTS, dtype=np.float32)
        one_hot[idx] = 1.0
        node_features.append(one_hot)
    x = torch.tensor(np.array(node_features), dtype=torch.float)

    # --- Edges from distance cutoff ---
    # Compute full distance matrix
    dist_matrix = squareform(pdist(positions))  # (N, N)

    src_list, dst_list, dist_list = [], [], []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            d = dist_matrix[i, j]
            if d <= cutoff:
                src_list.append(i)
                dst_list.append(j)
                dist_list.append(d)

    # Optionally limit neighbors
    if max_neighbors is not None and max_neighbors > 0:
        # For each node keep only the closest max_neighbors
        from collections import defaultdict
        neighbor_data = defaultdict(list)
        for idx_e in range(len(src_list)):
            neighbor_data[src_list[idx_e]].append(
                (dist_list[idx_e], dst_list[idx_e])
            )
        src_filtered, dst_filtered, dist_filtered = [], [], []
        for i in range(n_atoms):
            neighbors = sorted(neighbor_data[i], key=lambda x: x[0])
            for d, j in neighbors[:max_neighbors]:
                src_filtered.append(i)
                dst_filtered.append(j)
                dist_filtered.append(d)
        src_list, dst_list, dist_list = src_filtered, dst_filtered, dist_filtered

    if len(src_list) == 0:
        # If no edges within cutoff, connect to nearest neighbor at least
        for i in range(n_atoms):
            dists_i = dist_matrix[i].copy()
            dists_i[i] = np.inf
            j = int(np.argmin(dists_i))
            src_list.append(i)
            dst_list.append(j)
            dist_list.append(dist_matrix[i, j])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_dist = torch.tensor(dist_list, dtype=torch.float).unsqueeze(-1)  # (E, 1)

    # --- Gaussian RBF expansion of distances ---
    edge_attr = gaussian_rbf(edge_dist.squeeze(-1))  # (E, n_gaussians)

    # --- Target ---
    y = None
    if target_key is not None and target_key in atoms.info:
        y = torch.tensor([atoms.info[target_key]], dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_dist=edge_dist,
        y=y,
        n_atoms=torch.tensor([n_atoms], dtype=torch.long),
    )

    return data


def gaussian_rbf(
    distances: torch.Tensor,
    min_dist: float = 0.0,
    max_dist: float = 6.0,
    n_gaussians: int = 64,
) -> torch.Tensor:
    """
    Expand scalar distances into a Gaussian radial basis.

    Parameters
    ----------
    distances : (E,) tensor
    min_dist, max_dist : float
        Range of Gaussian centers.
    n_gaussians : int
        Number of Gaussian basis functions.

    Returns
    -------
    (E, n_gaussians) tensor
    """
    centers = torch.linspace(min_dist, max_dist, n_gaussians)  # (G,)
    gamma = 1.0 / ((max_dist - min_dist) / n_gaussians) ** 2
    # distances: (E,) -> (E, 1),  centers: (G,) -> (1, G)
    diff = distances.unsqueeze(-1) - centers.unsqueeze(0)
    return torch.exp(-gamma * diff ** 2)


class MoleculeDataset(Dataset):
    """
    Dataset that reads an extxyz file and converts each structure to a graph.
    """

    def __init__(
        self,
        xyz_file: str,
        cutoff: float = 5.0,
        target_key: str = 'bandgap_eV',
        max_neighbors: Optional[int] = None,
        n_gaussians: int = 64,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.target_key = target_key
        self.max_neighbors = max_neighbors
        self.n_gaussians = n_gaussians

        # Read all structures
        print(f"Reading structures from {xyz_file} ...")
        self.atoms_list = read(xyz_file, index=':')
        if not isinstance(self.atoms_list, list):
            self.atoms_list = [self.atoms_list]
        print(f"  Found {len(self.atoms_list)} structures.")

        # Pre-convert to graphs for speed
        self.graphs = []
        n_skipped = 0
        for atoms in self.atoms_list:
            try:
                g = atoms_to_graph(
                    atoms,
                    cutoff=self.cutoff,
                    target_key=self.target_key,
                    max_neighbors=self.max_neighbors,
                )
                self.graphs.append(g)
            except Exception as e:
                n_skipped += 1
                print(f"  Skipping structure: {e}")
        if n_skipped:
            print(f"  Skipped {n_skipped} structures due to errors.")

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx: int) -> Data:
        return self.graphs[idx]


def load_datasets(
    train_file: str,
    val_file: str,
    test_file: str,
    cutoff: float = 5.0,
    target_key: str = 'bandgap_eV',
    batch_size: int = 32,
    max_neighbors: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Load train/val/test extxyz files and return DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, stats
        stats contains 'mean' and 'std' of the training targets for normalization.
    """
    train_ds = MoleculeDataset(train_file, cutoff, target_key, max_neighbors)
    val_ds = MoleculeDataset(val_file, cutoff, target_key, max_neighbors)
    test_ds = MoleculeDataset(test_file, cutoff, target_key, max_neighbors)

    # Compute training set target statistics for normalization
    train_targets = torch.cat([g.y for g in train_ds.graphs if g.y is not None])
    stats = {
        'mean': train_targets.mean().item(),
        'std': train_targets.std().item(),
    }
    print(f"Training target stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, stats