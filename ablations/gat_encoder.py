from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.typing import Adj
from typing import Optional


class GATEncoder(nn.Module):
    """
    GATv2 (Brody et al., 2022) encoder.
    Tests whether CA-DGN's Antisymmetric ODE structure and directed RoPE aggregation
    add value over an attention-based baseline.

    GATv2Conv respects edge directionality (src attends to dst assymetrically).

    """

    def __init__(
            self,
            hidden_dim: int,
            num_layers: int = 5,
            num_heads: int = 4,
            dropout: float= 0.0,
    ):
        super(GATEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        head_dim = hidden_dim // num_heads

        # Weight-tied across layers (cf. CADGNEncoder's shared Conv approach)
        self.conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=num_heads,
            droupout=dropout,
            concat=True,  # output: (num_nodes, num_heads * head_dim)
            add_self_loops=True,
            bias = True
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
            self,
            x: torch.Tensor,  # (num_nodes, hidden_dim)
            edge_index: Adj,  # (2, num_edges)
            batch: Optional[torch.Tensor],  # unused, for compat
            diameters: Optional[torch.Tensor],  # unused, for compat
    ) -> torch.Tensor:
        # NB: GATv2Conv operates over a fixed number of layers, cf. CADGNEncoder which has dynamic depth.
        for _ in range(self.num_layers):
            x = self.conv(x, edge_index)
            x = self.dropout(x)
        return self.norm(x)
