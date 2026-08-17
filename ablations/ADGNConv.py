from torch_geometric.nn import GCNConv, AntiSymmetricConv
import torch
import torch.nn as nn


class ADGNConvWrapper(nn.Module):
    """
    Acts as a wrapper around AntiSymmetricConv (base ADGN implementation) - to bridge the API difference between CADGN and ADGN
    Gravina et al. (2022) A-DGN Neighbourhood aggregation.
    Implements Eq. 7 (GCN) due to Table 2 of Gravina et al. (ICLR 2023)
    suggesting its superior performance to the alternative A-DGN aggregation mechanism.
    """

    def __init__(
            self,
            hidden_dim: int,
            num_iters: int = 1,
            epsilon: float = 0.1,
            base_gamma: float = 0.1,
            act: str = "tanh"
    ):
        super(ADGNConvWrapper, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv = AntiSymmetricConv(
            in_channels=hidden_dim,
            phi=None,
            num_iters=num_iters,
            epsilon=epsilon,
            gamma=base_gamma,
            act=act
        )

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            layer: int = 1,
    ) -> torch.Tensor:
        """

        :param x: (num_nodes, hidden_dim) tensor
        :param edge_index: (2, num_edges)
        :param layer: to adopt standard CADGN foward() signature; this is ignored
        :return: (num_nodes, hidden_dim) tensor
        """
        return self.conv(x, edge_index)

