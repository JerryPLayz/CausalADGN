import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphBuilder(nn.Module):
    """
    Inner product decoder with a learned scaling MLP
    Used to reconstruct edges in a pairwise manner.
    """

    def __init__(self, hidden_dim: int):
        super(GraphBuilder, self).__init__()
        self.hidden_dim = hidden_dim
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self,
                Z: torch.Tensor,
                candidate_pairs: torch.Tensor
                ) -> torch.Tensor:

        src = Z[candidate_pairs[0]]  # (num_candidates, ca_dgn_dim)
        dst = Z[candidate_pairs[1]]  # (num_candidates, ca_dgn_dim)
        return self.score(torch.cat([src, dst], dim=-1)).squeeze(-1)

