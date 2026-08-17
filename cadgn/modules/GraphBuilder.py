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
            nn.Linear(hidden_dim, 1),
            # DO NOT ADD nn.Sigmoid here: F.binary_cross_entropy_with_logits applies log-sigmoid during eval.
        )

    def forward(self,
                Z: torch.Tensor,
                candidate_pairs: torch.Tensor
                ) -> torch.Tensor:

        src = Z[candidate_pairs[0]]  # (num_candidates, ca_dgn_dim)
        dst = Z[candidate_pairs[1]]  # (num_candidates, ca_dgn_dim)
        return self.score(torch.cat([src, dst], dim=-1)).squeeze(-1)

    @staticmethod
    def to_probs(
            edge_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies sigmoid function to edge logits
        :param edge_logits: raw edge logits from forward()
        :return:
        """
        return torch.sigmoid(edge_logits)

    @staticmethod
    def to_edges(
            edge_logits: torch.Tensor,
            candidate_pairs: torch.Tensor,
            threshold: float = 0.5
    ) -> torch.Tensor:
        probs = torch.sigmoid(edge_logits)
        mask = probs >= threshold
        return candidate_pairs[:, mask]

