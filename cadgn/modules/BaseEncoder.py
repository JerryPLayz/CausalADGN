from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.typing import Adj
from abc import ABC
from cadgn.modules.graph_utils import batch_diameter
from typing import Optional, Callable


class BaseEncoder(nn.Module):
    """
    Common base for all Encoders (e.g. CADGNEncoder, GATEncoder etc.)
    Handles common logic to ensure parity
    """
    def __init__(
            self,
            hidden_dim: int,
            max_layers: Optional[int],
            dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_layers = max_layers
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)

        self._conv = nn.Identity()

    @property
    def conv(self) -> nn.Module:
        return self._conv

    @conv.setter
    def conv(self, value: nn.Module) -> None:
        self._conv = value

    def run_conv(
            self,
            x: torch.Tensor,
            edge_index: Adj,
            layer: int,
            **kwargs
    ) -> torch.Tensor:
        """
        A simple helper to promote dynamicism, in order to minimize code duplication.
        Override this method if your Encoder needs additional parameters in the forward()
        This is run as shown in forward() in the loop across layers.
        :return: torch.Tensor - next value of x
        """
        return self.conv(x, edge_index)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(
            self,
            x: torch.Tensor,  # (num_nodes, hidden_dim)
            edge_index: Adj,  # (2, num_edges)
            batch: Optional[torch.Tensor] = None,  # (num_nodes, ) PyG batch vector; None= single graph
            diameters: Optional[torch.Tensor] = None,  # (num_graphs, ) precomputed (should be)
            **kwargs
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

        for layer_idx in range(num_layers):
            x = self.run_conv(
                x=x,
                edge_index=edge_index,
                layer=layer_idx+1,
                **kwargs)
            x = self.dropout(x)
        return self.norm(x)
