import torch
import torch.nn as nn
import math
import numpy as np
from typing import Any, Callable, Dict, Optional, Union

from RoPEDirectedNeighbourhoodAggregation import RoPEDirectedNeighborhoodAggregation
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj


class CADGNConv(nn.Module):
    """
    Single CA-DGN convolution layer implementing:

    h_u^(l+1) = h_u^(l) + epsilon * sigma*((W - W^T - gamma(G)*I)h_u^(l)  + Psi_c(u) - Psi_e(u) + 1[u in a(u)](W_sl - W_sl^T)h_u^(l) + b)
                                                         ^Stability   Cause Nbhd^      Eff N^        ^ Self Loop Contribution

    W - W^T:     The core Skew-Symmetric element of A-DGN, purely imaginary eigenvalues.

    """
    # TODO: Add in Self-Loop Contribution

    def __init__(
            self,
            hidden_dim: int,
            num_iters: int = 1,  # Euler-step coarseness
            epsilon: float = 0.1,
            base_gamma: float = 0.1,
            act: Union[str, Callable, None] = 'tanh',
            act_kwargs: Optional[Dict[str, Any]] = None,
            bias: bool = True
    ):
        super(CADGNConv, self).__init__()
        assert hidden_dim > 0, f"hidden_dim must be > 0. Got {hidden_dim=}."
        assert num_iters > 0, f"num_iters must be > 0. Got {num_iters=}."
        self.num_iters = num_iters
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.base_gamma = base_gamma

        # Free Weight, skew-symmetric property is calculated at runtime.
        self.W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.register_buffer('eye', torch.eye(hidden_dim))  # mirror A-DGN here.

        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_parameter('bias', None)

        # Graph Adaptive Gamma needs to be properly set, but for now, we can hardcode it.
        # TODO: make dynamic
        self.gamma_adapt = nn.Parameter(torch.zeros(1))

        # Directed RoPE neighbourhood aggregation : replaces Phi in A-DGN
        self.phi = RoPEDirectedNeighborhoodAggregation(hidden_dim)

        activations = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "gelu": F.gelu,
            None: None,
        }
        self.act = activation_resolver(act, **(act_kwargs or {}))

        self.reset_parameters()

    def reset_parameters(self):
        r""" Resets all learnable parameters of the module"""
        # implemented equiv to A-DGN
        torch.nn.init.kaiming_normal_(self.W, a=math.sqrt(5))
        self.phi.reset_parameters()
        zeros(self.bias)

    def forward(self,
                x: torch.Tensor,  # (num_nodes, hidden_dim)
                edge_index: Adj,  # (2, num_edges
                layer: int = 1
                ) -> torch.Tensor:
        ## todo: dynamicize gamma
        gamma = self.base_gamma + F.softplus(self.gamma_adapt)

        # Skew-symmetric Weight Matrix
        W_asym = self.W - self.W.t() - gamma * self.eye


        # Directed RoPE Neighbourhood Aggregation
        phi_out = self.phi(x, edge_index, layer)

        # Forward Euler Discretisation Step: higher self.num_iters = finer detail
        # NB: layer is fixed through all iters.
        for _ in range(self.num_iters):
            h = self.phi(x, edge_index=edge_index, layer=layer)
            h = x @ W_asym.t() + h

            if self.bias is not None:
                h = h + self.bias

            if self.act is not None:
                h = self.act(h)
            x = x + self.epsilon * h

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'{self.hidden_dim}, '
                f'num_iters={self.num_iters}, '
                f'epsilon={self.epsilon}, '
                f'base_gamma={self.base_gamma})')
