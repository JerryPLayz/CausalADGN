import torch
import torch.nn as nn
import math
import numpy as np
from typing import Any, Callable, Dict, Optional, Union
from torch_geometric.typing import Adj

from .CADGNConv import CADGNConv
from .graph_utils import *
from .BaseEncoder import BaseEncoder


class CADGNEncoder(BaseEncoder):
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
        super(CADGNEncoder, self).__init__(
            hidden_dim = hidden_dim,
            max_layers = max_layers,
            dropout=dropout
        )

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

    def run_conv(
            self,
            x: torch.Tensor,
            edge_index: Adj,
            layer: int,
            **kwargs
    ) -> torch.Tensor:
        return self.conv(
            x=x,
            edge_index=edge_index,
            layer=layer  # +1 occurs in forward()
        )



class CADGNDecoder(nn.Module):
    """
    Shared per-node MLP decoder between the CADGNEncoder and all DecoderHeads.
    Operates entirely in ca_dgn_dim.
    Each node is processed identically and independently.
    """
    def __init__(self,
                 ca_dgn_dim,
                 expansion: int=4,
                 dropout: float = 0.1
                 ):
        """
        :param ca_dgn_dim: The hidden dimension in which ca_dgn_dim operates in.
        :param expansion: Feed-Forward Network Expansion Factor.
        :param dropout: Dropout percentage for training.
        """
        super(CADGNDecoder, self).__init__()
        self.norm = nn.LayerNorm(ca_dgn_dim)
        self.ffn = nn.Sequential(
            nn.Linear(ca_dgn_dim, ca_dgn_dim * expansion),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ca_dgn_dim * expansion, ca_dgn_dim),
            nn.Dropout(p=dropout),
        )

        self.out_norm = nn.LayerNorm(ca_dgn_dim)
        self._init_weights()

    def forward(self, Z):
        """

        :param Z: (num_nodes, ca_dgn_dim) Neighbourhood-aware per-node embeddings from CADGNEncoder.
        :return: (num_nodes, ca_dgn_dim) Enriched per-node embeddings, attempting to universally transform all node embeddings to promote reconstruction of the original.
        """
        return self.out_norm(Z + self.ffn(self.norm(Z)))

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


