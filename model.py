"""
Message Passing Neural Network for scalar molecular property prediction.

Architecture
------------
1. Node embedding: linear projection of one-hot atomic features.
2. Two message-passing layers where messages depend on sender node features
   AND edge (distance) features. This keeps everything rotationally invariant.
3. Global mean pooling over all nodes → graph-level representation.
   Mean pooling makes the output invariant to the number of atoms (intensive).
4. Readout MLP maps the graph vector to a scalar prediction.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data


class MPNNLayer(MessagePassing):
    """
    A single message-passing layer.

    message:  m_{ij} = MLP_msg( [h_j || e_{ij}] )
    aggregate: agg_i = sum_j m_{ij}
    update:   h_i' = MLP_upd( [h_i || agg_i] )
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr='add')  # sum aggregation

        # Message function: takes concatenation of source node feat + edge feat
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Update function: takes concatenation of current node feat + aggregated msg
        self.upd_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),  # output same dim as input node feat
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        # propagate calls message -> aggregate -> update internally
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # Update
        out = self.upd_mlp(torch.cat([x, agg], dim=-1))
        # Residual connection + layer norm
        out = self.layer_norm(x + out)
        return out

    def message(self, x_j, edge_attr):
        # x_j: features of source nodes (neighbors), shape (E, node_dim)
        # edge_attr: shape (E, edge_dim)
        inp = torch.cat([x_j, edge_attr], dim=-1)
        return self.msg_mlp(inp)


class MPNN(nn.Module):
    """
    Full MPNN model:
        embed -> MP layer 1 -> MP layer 2 -> global mean pool -> readout MLP -> scalar
    """

    def __init__(
        self,
        n_atom_features: int = 36,   # one-hot dimension (number of supported elements)
        n_edge_features: int = 64,   # Gaussian RBF dimension
        node_dim: int = 128,
        hidden_dim: int = 128,
        n_mp_layers: int = 2,
        readout_hidden: int = 64,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        super().__init__()

        self.target_mean = target_mean
        self.target_std = target_std

        # --- Node embedding ---
        self.node_embed = nn.Sequential(
            nn.Linear(n_atom_features, node_dim),
            nn.SiLU(),
        )

        # --- Edge embedding (optional extra projection of RBF) ---
        self.edge_embed = nn.Sequential(
            nn.Linear(n_edge_features, hidden_dim),
            nn.SiLU(),
        )
        edge_dim_internal = hidden_dim

        # --- Message Passing Layers ---
        self.mp_layers = nn.ModuleList()
        for _ in range(n_mp_layers):
            self.mp_layers.append(
                MPNNLayer(
                    node_dim=node_dim,
                    edge_dim=edge_dim_internal,
                    hidden_dim=hidden_dim,
                )
            )

        # --- Readout MLP ---
        self.readout = nn.Sequential(
            nn.Linear(node_dim, readout_hidden),
            nn.SiLU(),
            nn.Linear(readout_hidden, readout_hidden),
            nn.SiLU(),
            nn.Linear(readout_hidden, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch_geometric.data.Data (or Batch)
            Must have: x, edge_index, edge_attr, batch

        Returns
        -------
        predictions : (B,) tensor of predicted band gaps (in original units, un-normalized).
        """
        x = data.x                     # (N_total, n_atom_features)
        edge_index = data.edge_index   # (2, E_total)
        edge_attr = data.edge_attr     # (E_total, n_edge_features)
        batch = data.batch             # (N_total,)

        # Embed
        h = self.node_embed(x)                  # (N_total, node_dim)
        e = self.edge_embed(edge_attr)           # (E_total, edge_dim_internal)

        # Message passing
        for mp in self.mp_layers:
            h = mp(h, edge_index, e)

        # Global mean pooling → intensive (size-independent) representation
        graph_vec = global_mean_pool(h, batch)   # (B, node_dim)

        # Readout
        pred_normalized = self.readout(graph_vec).squeeze(-1)  # (B,)

        # De-normalize
        pred = pred_normalized * self.target_std + self.target_mean

        return pred

    def predict_normalized(self, data: Data) -> torch.Tensor:
        """Return prediction in normalized (zero-mean unit-var) space."""
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)
        for mp in self.mp_layers:
            h = mp(h, edge_index, e)
        graph_vec = global_mean_pool(h, batch)
        return self.readout(graph_vec).squeeze(-1)