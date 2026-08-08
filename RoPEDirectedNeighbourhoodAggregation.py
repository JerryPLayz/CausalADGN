from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops
import torch
import torch.nn as nn
import math
import numpy as np


class RoPEDirectedNeighborhoodAggregation(MessagePassing):
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
        super(RoPEDirectedNeighborhoodAggregation, self).__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.base = base
        assert hidden_dim % 2 == 0, f"hidden_dim must be even for RoPE pairing, got {hidden_dim}"

        ## Parameters
        # Learnable direction-specific scalers
        self.alpha_cause = nn.Parameter(torch.ones(1))
        self.alpha_effect = nn.Parameter(torch.ones(1))

        # Fixed frequency indices - not learnable (consistent with RoFormer)
        i = torch.arange(0, hidden_dim // 2, dtype=torch.float32)
        self.register_buffer('freq_indices', i)

        self.W_self = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.W, a=math.sqrt(5))
        nn.init.ones_(self.alpha_cause)
        nn.init.ones_(self.alpha_effect)



    def debug_get_skew_symmetric_matrix(self, layer: int, alpha: torch.Tensor):
        """
        Explicitly constructs the block-diagonal skew-symmetric matrix \Omega^(l).
        Each 2x2 block along the diagonal is a skew-symmetric matrix:
            \Theta_i^(l) * J = [[0,              -\Theta_i^(l)],
                                [\Theta_i^(l),   0            ]]
        Satisfies: \Omega + \Omega^T = 0   (skew symmetry)
        Guarantees: exp(\Omega) is orthogonal

        Note: this method is for verification and logging only.
        apply_rotation() should be used in practice, rather than constructing this expensive matrix on every forward.
        :param layer:
        :param alpha:
        :return: Omega -> (hidden_dim, hidden_dim) skew-symmetric matrix
        """
        d = self.hidden_dim
        #   See RoFormer P3.3                   (-2 * i) / d
        theta_i = layer * alpha * (self.base ** (-2.0 * self.freq_indices / d)) ## TODO: check this formulation is correct
        Omega = torch.zeros(d,d, device=theta_i.device, dtype=theta_i.dtype)
        even_idx = torch.arange(0, d, 2, device=theta_i.device)  # [0, 2, 4, ..., d-2]
        odd_idx = torch.arange(1, d, 2, device=theta_i.device)   # [1, 3, 5, ..., d-1]

        # Upper Triangle: -theta_i at position (2i, 2i+1)
        # Lower Triangle: theta_i at position (2i+1, 2i)
        # Together, these enforce Omega = -Omega^T exactly.
        Omega[even_idx, odd_idx] = -theta_i   # positions (0,1), (2,3), ... -> -theta_i
        Omega[odd_idx, even_idx] = theta_i    # positions (1,0), (3,2), ... -> theta_i

        # NB: use Omega in exp(Omega) => ( (cos theta_i,  -sin theta_i), (sin theta_i, cos theta_i))
        return Omega

    def verify_skew_symmetry(self, layer: int, skew_ok: float = 1e-6):
        """
        A utility to call during debugging to confirm that skew-symmetry holds.
        :param layer: integer for the layer to test
        :return: dict
        """
        with torch.no_grad():
            Omega_c = self.debug_get_skew_symmetric_matrix(layer, alpha=self.alpha_cause)
            Omega_e = self.debug_get_skew_symmetric_matrix(layer, alpha=self.alpha_effect)
            err_c = (Omega_c + Omega_c.t()).abs().max().item()
            err_e = (Omega_e + Omega_e.t()).abs().max().item()
            return {
                "layer": layer,
                "cause_max_error": err_c,  # should be ~0.0
                "effect_max_error": err_e,  # should be ~0.0
                "cause_skew_ok": err_c < skew_ok,
                "effect_skew_ok": err_c < skew_ok,
            }

    def apply_rotation(self,
                       x: torch.Tensor,     # (num_nodes, hidden_dim)
                       layer: int,
                       alpha: torch.Tensor  # scalar nn.Parameter
                       ) -> torch.Tensor:
        """
        Applies R^(l) = exp(Omega^(l)) to each node embedding
        For a block-diagonal Omega with 2x2 blocks Theta_i * J, the matrix exponential resolves analytically per block.

        Follows the standard Rodrigues rotation formula in 2D. The result is identical to torch.linalg.matrix_exp(Omega)
          applied to x, but O(d) rather than O(d^3).
        :param x: The node embeddings
        :param layer: The current 'layer' of the model (e.g. discretisation)
        :param alpha:
        :return:
        """
        d = self.hidden_dim
        theta = layer * alpha * (self.base ** (-2.0 * self.freq_indices / d))   # (d//2)

        cos = torch.cos(theta)  # (d//2)
        sin = torch.sin(theta)  # (d//2)

        x1 = x[:, 0::2]  # (num_nodes, d//2) - even-indexed dimensions
        x2 = x[:, 1::2]  # (num_nodes, d//2) - odd-indexed dimensions

        # Block-wise application of exp(Omega^(l))
        x_rot = torch.stack(
            [
                x1 * cos - x2 * sin,
                x1 * sin + x2 * cos
            ],
            dim=-1
        ).flatten(-2)  # (num_nodes, hidden_dim)

        return x_rot

    def forward(self,
                x: torch.Tensor,  # (num_nodes, hidden_dim)
                edge_index: torch.Tensor,  # (2, num_edges) src -> dst means src causes dst
                layer: int = 1
                ) -> torch.Tensor:
        """
        Returns Psi_c(u) - Psi_e(u) + Psi_self for all nodes u simultaneously.
        Psi_c(u) aggregates over all nodes that CAUSE u (nodes where edge v->u exists, incoming to u)
          After Rotation by R_cause^(l) - No Self-Loops

        Psi_e(u) aggregates over all nodes that u CAUSES (nodes v where u->v exists, outgoing from u)
          After Rotation by R_effect^(l) - No Self-Loops

        Psi_self adds to the embedding if and only if the node has a self-loop (that is `(i,i)∈ℰ`  for any `i∈V`)

        edge_index convention:
          edge_index[0] = source (cause)
          edge_index[1] = target (effect)
          i.e. edge_index[0,k] -> edge_index[1, k]
        :param x:
        :param edge_index:
        :param layer:
        :return:
        """
        # Ensure that no self-loop contribution gets added to cause/effect (only via W_self)
        self_loop_mask = edge_index[0] == edge_index[1]  # (num_edges, ) bool
        self_loop_nodes = edge_index[0, self_loop_mask]  # node indices with self-loops
        edge_index_no_self, _ = remove_self_loops(edge_index)

        # Psi_cause = incoming neighbors only (no self-loops)
        x_cause_rot = self.apply_rotation(x, layer, self.alpha_cause)
        psi_c = self.propagate(edge_index_no_self, x=x_cause_rot)

        # Psi_effect = outgoing neighbors only (no self-loops)
        x_effect_rot = self.apply_rotation(x, layer, self.alpha_effect)
        edge_index_rev = edge_index_no_self.flip(0)
        psi_e = self.propagate(edge_index_rev, x=x_effect_rot)

        # Self-loop contribution exists where i=j (along diagonal only), only for nodes that have them.
        #  W - W^T antisymmetry again
        W_self_asym = self.W_self - self.W_self.t()
        psi_self = torch.zeros_like(x)  # make empty tensor of shape X (zeros everywhere)
        psi_self[self_loop_nodes] = x[self_loop_nodes] @ W_self_asym.t()  # only replace values for nodes that have self-loops

        return psi_c - psi_e + psi_self

    def message(self, x_j: torch.Tensor):
        # No additional transformation needed here (rotation will be applied out of scope)
        return x_j

