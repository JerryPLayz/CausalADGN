from __future__ import annotations
from itertools import combinations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# # Mean-Maximum Discrepancy Error # #


# Kernels for Testing
def _imq_kernel(
        X: torch.Tensor,
        Y: torch.Tensor,
        c: float,
        beta: float
) -> torch.Tensor:
    """
    Inverse-Multiquadratic Kernel between tensors X and Y
    k_IMQ(x,y) = (c^2 / (c^2  + ||x-y||^2) )^beta
    :param X: :class:`~torch.Tensor` of shape (num_nodes, hidden_dim)
    :param Y: :class:`~torch.Tensor` of shape (num_nodes, hidden_dim)
    :param c: Scale parameter, analogous to sigma in RBF kernel
    :param beta: Decay exponent in range (0,1), controls tail heaviness
    :return: :class:`~torch.Tensor` of shape (num_nodes, num_nodes)
    """
    diff = X.unsqueeze(1) - Y.unsqueeze(0)  # (n, m, d)
    sq = diff.pow(2).sum(dim=-1)  # (n, m)
    return (c ** 2 / (c ** 2 + sq)) ** beta


def _rbf_kernel(
        X: torch.Tensor,
        Y: torch.Tensor,
        sigma: float
):
    """
    Radial Basis Function (Gaussian) Kernel
    k_RBF(x, y) = exp(- ||x - y||^2 / (2 * sigma^2))

    :param X: :class:`~torch.Tensor` of shape (num_nodes, hidden_dim)
    :param Y: :class:`~torch.Tensor` of shape (num_nodes, hidden_dim)
    :param sigma: pre-computed shared bandwidth (median heuristics over all N distributions)
    :return: :class:`~torch.Tensor` of shape (num_nodes, num_nodes)
    """
    diff = X.unsqueeze(1) - Y.unsqueeze(0)  # (n, m, d)
    sq = diff.pow(2).sum(-1)  # (n, m)
    return torch.exp(-sq / (2.0 * sigma ** 2))


def _mmd_pair(
        Z_a: torch.Tensor,
        Z_b: torch.Tensor,
        bandwidth: float,
        beta: Optional[float] = None,
        kernel: Literal["rbf", "imq"] = "imq",
        *args, **kwargs
) -> torch.Tensor:
    r"""
    An Unbiased MMD^2 calculation between two tensors using either rbf or imq kernels.
    :param Z_a: :class:`~torch.Tensor` of shape (num_nodes, hidden_dim)
    :param Z_b: :class:`~torch.Tensor` of shape (num_nodes, hidden_dim)
    :param bandwidth: Used in both RBF (sigma) and IMQ kernels (c), estimates scale for both.
    :param beta: Decay exponent, used only in the IMQ kernel. Must be in range (0, 1), controlling tail heaviness.
    :param kernel: string in ['rbf', 'imq'] which dictates which kernel should be used to calculate MMD loss.
    :return: Unbiased MMD^2 score of Z_a and Z_b.
    """

    if kernel == "rbf":
        K_aa = _rbf_kernel(Z_a, Z_a, bandwidth).mean()
        K_bb = _rbf_kernel(Z_b, Z_b, bandwidth).mean()
        K_ab = _rbf_kernel(Z_a, Z_b, bandwidth).mean()
    else:  # imq
        K_aa = _imq_kernel(Z_a, Z_a, bandwidth, beta).mean()
        K_bb = _imq_kernel(Z_b, Z_b, bandwidth, beta).mean()
        K_ab = _imq_kernel(Z_a, Z_b, bandwidth, beta).mean()

    return K_aa + K_bb - 2 * K_ab


def generalized_mmd_loss(
        Z_dict: Dict[str, torch.Tensor],
        kernel: Literal["rbf", "imq"] = "imq",
        beta: float = 0.5
) -> torch.Tensor:
    """
    Generalized MMD loss across N tokenizer family embedding distributions.
    Computes the normalized average pairwise MMD^2 across all pairwise combinations.
    Bandwidth is estimated jointly from all N distributions via the median heuristic, ensuring consistent bandwidth across all pairs.

    For N=2 it is equivalent to a simplified pairwise implementation.
    :param Z_dict: mapping of family_name -> Z tensor (num_nodes, hidden_dim) All Z tensors must be on the same device
    :param kernel: string in ['rbf', 'imq'] which dictates which kernel should be used to calculate MMD loss.
    :param beta: IMQ decay exponent in range (0, 1), controlling tail heaviness, default 0.5 (standard IMQ). Only used when kernel="imq". Ignored for RBF kernel.
    :return: Scalar MMD loss tensor, mean over all pairs.
    """
    Z_list = list(Z_dict.values())
    N = len(Z_list)

    if N < 2:
        # Single-family, no alignment possible
        return torch.tensor(0.0, device=Z_list[0].device, requires_grad=False)

    assert 0.0 < beta < 1.0 or kernel == "rbf", (
        f"IMQ beta must be in (0, 1). Got {beta}."
    )


    # Bandwidth Estimation (shared)
    # For RBF, bandwidth = sigma (std dev parameter)
    # For IMQ, bandwidth = c (scale parameter, same geometric meaning)
    with torch.no_grad():
        all_Z = torch.cat(Z_list, dim=0)
        bandwidth = (
            torch.cdist(all_Z, all_Z)
            .median()
            .clamp(min=1e-3)
            .item()
        )

    # All pairwise MMDs
    pair_indices = list(combinations(range(N), 2))  # C(N, 2) pairs
    pair_losses = [
        _mmd_pair(
            Z_list[i],
            Z_list[j],
            bandwidth=bandwidth,
            kernel=kernel,
            beta=beta,

        )
        for i, j in pair_indices
    ]

    # Normalize by number of pairsto ensure consistent scale regardless of N
    return sum(pair_losses) / len(pair_losses)


# # Edge Loss # #
def _generate_candidate_pairs(
        num_nodes: int,
        batch_vector: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Generate all valid directed candidate pairs for edge prediction (reconstruction)
    For a single graph, this will generate all N*(N-1) directed pairs (no self-loops - yet. TODO )
    For a batched graph (batch_vector provided): generates pairs only within each component graph.
    :param num_nodes: Total Node Count; Z.size(0) across all graphs in batch
    :param batch_vector: (num_nodes, ) LongTensor mapping each node to a graph index. None implies a single graph.
    :param device: Target Device. Inferred from batch_vector if None.
    :return: candidate_pairs in shape (2, num_candidates) LongTensor. Row 0: source (cause) node indices;; Row 2: destination (effect) node indices. All indices are global (consistent with batched edge_index)
    """

    if device is None and batch_vector is not None:
        device = batch_vector.device

    if device is None:
        device = torch.device("cpu")

    # Single Graph
    if batch_vector is None:
        src = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
        dst = torch.arange(num_nodes, device=device).repeat(num_nodes)
        mask = src != dst # todo: remove when self-loops are permitted
        return torch.stack(src[mask], dst[mask])

    # Batched graphs: pairwise only within same graph
    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []

    for g_idx in batch_vector.unique(sorted=True):
        # Global node indices for this graph
        nodes = (batch_vector == g_idx).nonzero(as_tuple=True)[0]
        n_g = len(nodes)

        if n_g < 2:
            # Single node graph, no valid pairs
            continue
        # All directed pairs within this graph (using global indices)
        i = nodes.repeat_interleave(n_g)  # (n_g^2, )
        j = nones.repeat(n_g)  # (n_g^2, )
        within_mask = i != j  # todo: remove when self-loops are permitted
        src_parts.append(i[within_mask])
        dst_parts.append(j[within_mask])

    if not src_parts:
        # All single-node graphs
        return torch.zeros(2, 0, dtype=torch.long, device=device)

    return torch.stack([
        torch.cat(src_parts),
        torch.cat(dst_parts)
    ])


def edge_reconstruction_loss(
        edge_logits: torch.Tensor,
        candidate_pairs: torch.Tensor,
        true_edge_index: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None,
        auto_pos_weight: bool = True,
        reduction: str = "mean"
) -> torch.Tensor:
    """
    Binary cross-entropy loss for directed edge prediction.
    Measures how faithfully the GraphBuilder reconstructs directed causal adjacency from node embeddings.
    Full gradient: loss -> edge_logits -> GraphBuilder -> Z -> CADGNEncoder -> H -> EncoderHead

    :param edge_logits: tensor of shape (num_candidates,); raw GraphBuilder scores
    :param candidate_pairs:  tensor of shape (2, num_candidates); from _generate_candidate_pairs()
    :param true_edge_index:  tensor of shape (2, num_true_edges); ground truth directed edges
    :param pos_weight:  manual positive class weight, overrides auto if provided
    :param auto_pos_weight: Compute pos_weight automatically (recommended: True)
    :param reduction: the kind of reduction in which to produce the scalar - e.g. 'sum' or 'mean'
    :return: Scalar BCE loss, mean over all candidates
    """

    device = edge_logits.device

    if candidate_pairs.size(1) == 0:
        # No candidates (e.g. all single-node graphs), return zero loss
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Ground Truth is fully vectorized
    # Flat indexes map (src, dst) to a unique integer
    # max_node_id = max global node index in either set

    max_id = max(
        candidate_pairs.max().item(),
        true_edge_index.max().item() if true_edge_index.numel() > 0 else 0
    ) + 1

    flat_candidates = (
        candidate_pairs[0] * max_id + candidate_pairs[1]
    )  # (num_candidates, )

    flat_true = (
        true_edge_index[0] * max_id + true_edge_index[1]
    ) if true_edge_index.numel() > 0 else torch.empty(
        0, dtype=torch.long, device=device
    )  # (num_true_edges, )

    gt = torch.isin(flat_candidates, flat_true).float()
    # (num_candidates, ) - 1.0 for true edges, 0.0 otherwise

    if pos_weight is not None:
        weight = pos_weight.to(device=device)
    elif auto_pos_weight:
        num_pos = gt.sum().clamp(min=1.0)
        num_neg = (gt == 0.0).float().sum()
        weight = (num_neg / num_pos).unsqueeze(0)
    else:
        weight = None

    return F.binary_cross_entropy_with_logits(
        edge_logits,
        gt,
        pos_weight=weight,
        reduction=reduction
    )


# # Node Reconstruction Loss # #
def node_reconstruction_loss(
        token_logits: torch.Tensor,
        token_ids: torch.Tensor,
        pad_id: int,
        label_smoothing: float = 0.0,
        reduction: str = "mean"
) -> torch.Tensor:
    """
    Cross-Entropy reconstruction loss over fully tokenized sequences.
    Measures how faithfully the Decoder Head recovers the original token sequence for each node from its CA-DGN latent embedding.

    Full Gradient Flow: loss -> token_logits -> DecoderHead -> CADGNDecoder -> Z -> CADGNEncoder -> H -> EncoderHead
    :param token_logits: tensor of shape (num_nodes, seq_len, vocab_size); Raw logits from DecoderHead (not softmaxed)
    :param token_ids: tensor of shape (num_nodes, seq_len) LongTensor; Ground truth, same token_ids fed to Encoder Head
    :param pad_id: Token ID for padding, these positions are ignored; from `family.tokenizer.pad_token_id`
    :param label_smoothing: In range [0, 1). 0.0 = Hard Targets. Not recommended for Stage 1 where the reconstruction target should be precise. Consider 0.005-0.1 for Stage 2 where the reconstruction signal is a softer constraint.
    :param reduction: the kind of reduction in which to produce the scalar - e.g. 'sum' or 'mean'
    :return: Scalar cross-entropy; mean over all non-padding token positions.
    """

    N,S, V = token_logits.shape

    return F.cross_entropy(
        token_logits.reshape(N * S, V),  # (N*S, V)
        token_ids.reshape(N*S),  # (N * S, )
        ignore_index=pad_id,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


# # Embedding Norm Loss # #
def embedding_norm_loss(
        Z: torch.Tensor,
        H: torch.Tensor,
        epsilon: float = 1e-8,
        w_preserve: flaot = 1.0,
        w_floor: float = 1.0
) -> torch.Tensor:
    """
    Embedding Norm Regularization, complementing the architectural non-dissipativity from the A-DGN framework.

    Whilst the CA-DGN skewsymmetry ensures non-dissipativity in the ODE dynamics, the EncoderHead and DecoderHeads are unconstrained.
    Hence, both amplification and collapse can occur. This regularization loss function is split in two:
    L_preserve: penalizes deviation between CA-DGN output (Z) and input (H) norms. The encoder should transmit information without scaling it.
    L_floor: log barrier to prevent representation collapse

    When used in Stage 2, this can be applied to the output of the Post-Projector as well, to ensure the LLM does not dissipate structural information injected by the Pre-Projector.
    Gradient for Stage 1: loss -> Z -> CADGNEncoder -> EncoderHead
    Gradient for Stage 2: loss -> Z_reconstructed -> PostProjector -> LLM activations -> PreProjector


    :param Z: tensor of shape (num_nodes, hidden_dim); gradient flows through this param
    :param H: tensor of shape (num_nodes, hidden_dim); norm reference, detached internally
    :param epsilon: Numerical floor for log, prevents log(0) in edge cases
    :param w_preserve: Weight on norm preservation term
    :param w_floor: Weight on log barrier term, increase if collapse persists, decrease if norms grow.
    :return: Scalar loss; w_preserve * L_preserve + w_floor * L_floor
    """
    z_norms = Z.norm(dim=-1)  # (num_nodes, )
    h_norms = H.norm(dim=-1).detach()  # (num_nodes, ), for reference

    L_preserve = F.mse_loss(z_norms, h_norms)
    L_floor = -torch.log(z_norms.clamp(min=epsilon)).mean()
    return w_preserve * L_preserve + w_floor * L_floor


# # Convenient Wrapper for all losses above # #
# Stage 1
def reconstruction_loss(
        token_logits: torch.Tensor,
        token_ids: torch.Tensor,
        pad_id: int,
        edge_logits: torch.Tensor,
        candidate_pairs: torch.Tensor,
        true_edge_index: torch.Tensor,
        Z: torch.Tensor,
        H: torch.Tensor,
        w_nodes: float = 1.0,
        w_edges: float = 1.0,
        w_norm: float = 0.01,
        label_smoothing: float = 0.0,
        auto_pos_weight: bool = True,
        norm_epsilon: float = 1e-8,
        norm_w_preserve: float = 1.0,
        norm_w_floor: float = 1.0,
        reduction: str = "mean"  # todo: check whether mean is appropriate here.
) -> dict[str, torch.Tensor]:
    """
    Computes all three losses for one family's forward pass. (Stage 1)
    Returns a dict so individual components are accessible for logging without recomputing. The top-level 'loss' key is the only tensor that is available for gradients.
    :param token_logits: tensor of shape (num_nodes, seq_len, vocab_size); Raw logits from DecoderHead (not softmaxed)
    :param token_ids: tensor of shape (num_nodes, seq_len) LongTensor; Ground truth, same token_ids fed to Encoder Head
    :param pad_id: Token ID for padding, these positions are ignored; from `family.tokenizer.pad_token_id`
    :param edge_logits: tensor of shape (num_candidates,); raw GraphBuilder scores
    :param candidate_pairs: tensor of shape (2, num_candidates); from _generate_candidate_pairs()
    :param true_edge_index: tensor of shape (2, num_true_edges); ground truth directed edges
    :param Z: tensor of shape (num_nodes, hidden_dim); gradient flows through this param
    :param H: tensor of shape (num_nodes, hidden_dim); norm reference, detached internally
    :param w_nodes: Overall weight for the node reconstruction loss of the loss function
    :param w_edges: Overall weight for the edge reconstruction loss of the loss function
    :param w_norm:  Overall weight for norm preservation loss of the loss function
    :param label_smoothing: For Node Reconstruction cross-entropy. In range [0, 1). 0.0 = Hard Targets. Not recommended for Stage 1 where the reconstruction target should be precise. Consider 0.005-0.1 for Stage 2 where the reconstruction signal is a softer constraint.
    :param auto_pos_weight: Compute pos_weight for edge BCE automatically (recommended: True)
    :param norm_epsilon: For embedding norm loss: Numerical floor for log, prevents log(0) in edge cases
    :param norm_w_preserve: For embedding norm loss: Weight on norm preservation term
    :param norm_w_floor: For embedding norm loss: Weight on log barrier term, increase if collapse persists, decrease if norms grow.
    :param reduction: the kind of reduction in which to produce the scalar - e.g. 'sum' or 'mean'
    :return: dict with keys: ['loss', 'L_nodes', 'L_edges', 'L_norm'], only 'loss' is gradient attached.
    """

    L_nodes = node_reconstruction_loss(
        token_logits=token_logits,
        token_ids=token_ids,
        pad_id=pad_id,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )

    L_edges = edge_reconstruction_loss(
        edge_logits=edge_logits,
        candidate_pairs=candidate_pairs,
        true_edge_index=true_edge_index,
        auto_pos_weight=auto_pos_weight,
        reduction=reduction,
    )

    L_norm = embedding_norm_loss(
        Z=Z,
        H=H,
        epsilon=norm_epsilon,
        w_preserve=norm_w_preserve,
        w_floor=norm_w_floor,
    )

    L_total = w_nodes * L_nodes + w_edges * L_edges + w_norm * L_norm

    return {
        'loss': L_total,
        'L_nodes': L_nodes.detach(),
        'L_edges': L_edges.detach(),
        'L_norm': L_norm.detach(),
    }


