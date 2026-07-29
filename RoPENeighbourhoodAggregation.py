from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn
import math
import numpy as np


class RoPENeighborhoodAggregation(MessagePassing):
    """
    Direction-specific neighborhood aggregation with monotonically increasing rotation to encode degree (based loosely on RoPE).
    Replaces the default *phi* in AntiSymmetricConv for directed graphs.
    Architecturally mirrors A-DGN's `W - W^T` construction, and thus is both skew-symmetric and orthogonal.
    Separates cause neighbors (incoming edges) from effect neighbors (outgoing edges), each with its own RoPE rotation matrix.

    Two separate learnable scalers, a_cause and a_effect, allow the model to learn distinct depth-encoding rates for each causal direction.
    """

    def __init__(
            self,
            hidden_dim: int,
            base: float = 10000.0):
        super(RoPENeighborhoodAggregation, self).__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.base = base

        ## Parameters
        # Learnable direction-specific scalers
        self.alpha_cause = nn.Parameter(torch.ones(1))
        self.alpha_effect = nn.Parameter(torch.ones(1))

        # Fixed frequency indices - not learnable (consistent with RoFormer)
        i = torch.arange(0, hidden_dim // 2, dtype=torch.float32)
        self.register_buffer('freq_indices', i)









