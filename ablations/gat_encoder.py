from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.typing import Adj
from typing import Optional
from cadgn import BaseEncoder


class GATEncoder(BaseEncoder):
    """
    GATv2 (Brody et al., 2022) encoder.
    Tests whether CA-DGN's Antisymmetric ODE structure and directed RoPE aggregation
    add value over an attention-based baseline.

    GATv2Conv respects edge directionality (src attends to dst assymetrically).

    """

    def __init__(
            self,
            hidden_dim: int,
            max_layers: int = 5,
            dropout: float= 0.0,
            num_heads: int = 4,
    ):
        super(GATEncoder, self).__init__(
            hidden_dim=hidden_dim,
            max_layers=max_layers,
            dropout=dropout
        )
        head_dim = hidden_dim // num_heads

        # Weight-tied across layers (cf. CADGNEncoder's shared Conv approach)
        self.conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=num_heads,
            # droupout=dropout,
            concat=True,  # output: (num_nodes, num_heads * head_dim)
            add_self_loops=True,
            bias = True
        )



