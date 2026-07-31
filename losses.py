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



# L2-Norm?