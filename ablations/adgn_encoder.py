from torch_geometric.nn import GCNConv, AntiSymmetricConv
import torch
import torch.nn as nn
from cadgn import BaseEncoder


class ADGNEncoder(BaseEncoder):
    """
    Acts as a wrapper around AntiSymmetricConv (base ADGN implementation) - to bridge the API difference between CADGN and ADGN
    Gravina et al. (2022) A-DGN Neighbourhood aggregation.
    Implements Eq. 7 (GCN) due to Table 2 of Gravina et al. (ICLR 2023)
    suggesting its superior performance to the alternative A-DGN aggregation mechanism.
    Also uses weight-tying for apples-to-apples comparisons.
    """

    def __init__(
            self,
            hidden_dim: int,
            max_layers: int = 5,
            dropout: float = 0.0,
            num_iters: int = 1,
            epsilon: float = 0.1,
            base_gamma: float = 0.1,
            act: str = "tanh"
    ):
        super(ADGNEncoder, self).__init__(
            hidden_dim=hidden_dim,
            max_layers=max_layers,
            dropout=dropout
        )
        self.conv = AntiSymmetricConv(
            in_channels=hidden_dim,
            phi=None,
            num_iters=num_iters,
            epsilon=epsilon,
            gamma=base_gamma,
            act=act
        )



