import torch
import torch.nn as nn
import math
import numpy as np
from .RoPEDirectedNeighbourhoodAggregation import RoPEDirectedNeighborhoodAggregation

class CADGNConv(nn.Module):
    """
    Single CA-DGN convolution layer implementing:

    h_u^(l+1) = h_u^(l) + epsilon * sigma*((W - W^T - gamma(G)*I)h_u^(l)  + Psi_c(u) - Psi_e(u) + 1[u in a(u)](W_sl - W_sl^T)h_u^(l) + b)
                                                         ^Stability   Cause Nbhd^      Eff N^        ^ Self Loop Contribution

    W - W^T:     The core Skew-Symmetric proposal of A-DGN, purely imaginary eigenvalues.

    """
    # TODO: Add in Self-Loop Contribution

    def __init__(self,
                 hidden_dim: int,
                 epsilon: float = 0.1,
                 base_gamma: float = 0.1,
                 act: str = "tanh",
                 bias: bool = True
                 ):
        super(CADGNConv, self).__init__()
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.base_gamma = base_gamma

        # Free Weight, skew-symmetric property is calculated at runtime.
        self.W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim)) if bias else None

        # Graph Adaptive Gamma needs to be properly set, but for now, we can hardcode it.
        # TODO: make dynamic
        self.gamma_adapt = nn.Parameter(torch.zeros(1))

        # Directed RoPE neighbourhood aggregation : replaces Phi in A-DGN
        self.phi = RoPEDirectedNeighborhoodAggregation(hidden_dim)

        activations = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "gelu": F.gelu,
        }
        assert act in activations, f"act must be one of {list(activations)}. Got {act}."
        self.act = activations[act]

        nn.init.kaiming_normal_(self.W, a=math.sqrt(5))


