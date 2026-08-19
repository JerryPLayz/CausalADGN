from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.typing import Adj

from cadgn.modules.graph_utils import batch_diameter
from cadgn import BaseEncoder
from typing import Optional


class GCNEncoder(BaseEncoder):
    """
    Standard GCN (Kipf & Welling, 2017) encoder.
    Tests whether CA-DGN's antisymmetry adds value over a standard message passing baseline (as compared to attentional)
    This implementation involves weight-tying to remain conceptually equivalent with other ablations and the CADGN framework.
    """

    def __init__(
            self,
            hidden_dim: int,
            max_layers: Optional[int],
            dropout: float = 0.0,
    ):
        super(GCNEncoder, self).__init__(
            hidden_dim=hidden_dim,
            max_layers=max_layers,
            dropout=dropout
        )
        # Weight-tied single conv
        self.conv = GCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            bias=True,
            add_self_loops=True,
        )

