from typing import Iterable

from graph_utils import GraphBatch, _RequiredBatchFields
from dataclasses import dataclass
from CADGNCore import CADGNCore
from TokenizerFamily import TokenizerFamily

import torch
import torch.nn as nn


@dataclass
class Stage1Config:
    """
    All hyperparameters for Stage 1 training. (grid search over these and the model hyperparameters as well)
    Preferred init via dict:
        config = Stage1Config(**param_grid_entry
    """
    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 1e-5

    # Training Loop
    epochs: int = 100
    grad_clip: float = 1.0  # 0.0 disables clipping

    # Top-Level Loss Weights
    w_nodes: float = 1.0   # node reconstruction
    w_edges: float = 1.0   # edge reconstruction
    w_norm : float = 0.01  # embedding norm preservation
    w_mmd  : float = 1.0   # cross-family MMD alignment for EncoderHeads & pre/post projectors

    # Node Reconstruction Sub-Weights
    w_mse  : float = 1.0
    w_cosine:float = 1.0

    # Norm Loss Sub-Weights
    norm_w_preserve: float = 1.0
    norm_w_floor   : float = 1.0
    norm_epsilon   : float = 1e-8

    # MMD
    mmd_kernel: str = 'imq'
    mmd_beta: float= 0.5

    # Edge Loss
    auto_pos_weight: bool = True
    reduction: str = 'mean'


@dataclass
class Stage2Config:
    """
    All Hyperparameters for Stage 2 training
    """
    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 1e-5

    # Training Loop
    epochs: int = 100
    grad_clip: float = 1.0  # 0.0 disables clipping

    reduction: str = 'mean'


class CADGNTrainer:
    """
    Orchestrates Stage 1 and Stage 2 training for the CA-DGN Framework
    Not an nn.Module.
    """
    def __init__(
            self,
            core: CADGNCore,
            families: Iterable[TokenizerFamily],
            device: str | torch.device = "cuda",
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.core: CADGNCore = core.to(device)
        self.families = list(families)
        for fam in self.families:
            fam.to(device)

    # Optimizer Construction
    def _build_stage1_optimizer(
            self,
            config: Stage1Config
    ) -> torch.optim.AdamW:
        """
        Joint optimizer over core + all TokenizerFamily Stage 1 parameters.
        Must be called after `configure_stage1()` on all components.
        :param config: The Config of Hyperparameters to use to train Stage 1.
        :return: AdamW Optimizer
        """
        params = list(self.core.trainable_parameters) + [
            p for fam in self.families for p in fam.trainable_parameters
        ]
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    def _build_stage2_optimizer(
            self,
            family: TokenizerFamily,
            config: Stage2Config
    ):
        """
        Optimizer scoped to one family's Stage 2 parameters only.
        Must be called after `family.configure_stage2()` and `core.configure_stage2()`
        :param config: The Stage 2 Config to use
        :return: AdamW Optimizer
        """
        return torch.optim.AdamW(
            list(family.trainable_parameters)
        )
