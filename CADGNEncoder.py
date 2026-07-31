import torch
import torch.nn as nn
import math
import numpy as np
from typing import Any, Callable, Dict, Optional, Union
from torch_geometric.typing import Adj

from CADGNConv import CADGNConv
from graph_utils import *


class CADGNEncoder(nn.Module):
    """
    Weight-tied CA-DGN Encoder with dynamic depth (based on input graph diameter).

    Dynamic depth:  dynamic_depth = diameter(G), or max over all graphs in batch
    Safety assured:   num_layers = min(dynamic_depth, max_layers)
    This prevents excessive compute on problematic graphs. If max_layers=None, then no cap is applied.

    The conv layer can be injected to allow:
        - Sharing across multiple encoders
        - Using a pre-trained conv
        - Custom subclasses of CADGNConv

    If conv=None, a default CADGNConv is created instead, using the remaining keyword arguments.
    """
    def __init__(
            self,
            hidden_dim: int,
            conv: Optional["CADGNConv"] = None,
            max_layers: Optional[int] = None,  # max receptive field depth
            num_iters: int = 1,   # Euler step discretisation - increase for finer discretisation per layer
            epsilon: float = 0.1,
            base_gamma: float = 0.1,
            act: Union[str, Callable, None] = 'tanh',
            dropout: float = 0.0,
            *args, **kwargs
    ):
        super(CADGNEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_layers = max_layers

        self.conv = conv if conv is not None else CADGNConv(
            hidden_dim=hidden_dim,
            num_iters=num_iters,
            epsilon=epsilon,
            base_gamma=base_gamma,
            act=act,
            *args,
            **kwargs
        )

        if conv is not None:
            assert hasattr(conv, 'hidden_dim'), "Injected Convolution must expose hidden_dim attribute"
            assert conv.hidden_dim == hidden_dim, f"Injected Convolution {conv.hidden_dim=} must match encoder hidden_dim {hidden_dim=}"

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
            self,
            x: torch.Tensor,  # (num_nodes, hidden_dim)
            edge_index: Adj,  # (2, num_edges)
            batch: Optional[torch.Tensor] = None,  # (num_nodes, ) PyG batch vector; None= single graph
            diameters: Optional[torch.Tensor] = None,  # (num_graphs, ) precomputed (should be)
    ) -> torch.Tensor:
        num_layers = batch_diameter(
            edge_index=edge_index,
            num_nodes=x.size(0),
            batch_vector=batch,
            precomputed=diameters
        )

        if self.max_layers is not None:
            num_layers = min(num_layers, self.max_layers)

        # Guard: always apply at least one layer
        num_layers = max(num_layers, 1)

        # Shared Convolution applied num_layer times
        for layer_idx in range(num_layers):
            x = self.conv(x, edge_index, layer=layer_idx + 1)
            x = self.dropout(x)

        return self.norm(x)
