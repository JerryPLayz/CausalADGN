"""
CADGNCore is a container for the shared CA-DGN framework modules

It holds the CADGNEncoder, CADGNDecoder and the GraphBuilder
All three modules are the shared core of CADGN across all LLM families.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from cadgn import CADGNEncoder, CADGNDecoder, GraphBuilder
from stager import Staged


class CADGNCore(nn.Module, Staged):
    SAVE_LOAD_PREFIX = "CADGNCore"
    def __init__(
            self,
            ca_dgn_dim: int,
            # CADGNEncoder
            max_layers: Optional[int] = None,
            num_iters: int = 1,
            epsilon: float = 0.1,
            base_gamma: float = 0.1,
            act: Union[str, None] = "tanh",
            encoder_dropout: float = 0.0,

            # CADGNDecoder
            decoder_expansion: int = 4,
            decoder_dropout: float = 0.1,

            # Graph Builder
    ):
        super().__init__()

        if callable(act) and not isinstance(act, str):
            raise ValueError(
                "`act` must be a string or None for CADGNCore to save/load properly. "
                "Callables cannot be serialized to config.json. "
            )

        self._config: dict = {
            "ca_dgn_dim": ca_dgn_dim,
            "max_layers": max_layers,
            "num_iters": num_iters,
            "epsilon": epsilon,
            "base_gamma": base_gamma,
            "act": act,
            "encoder_dropout": encoder_dropout,
            "decoder_expansion": decoder_expansion,
            "decoder_dropout": decoder_dropout,
        }

        # conv is not exposed; CADGNEncoder instantiates it by default.
        self.encoder = CADGNEncoder(
            hidden_dim=ca_dgn_dim,
            conv=None,
            max_layers=max_layers,
            num_iters=num_iters,
            epsilon=epsilon,
            base_gamma=base_gamma,
            act=act,
            dropout=encoder_dropout,
        )

        self.decoder = CADGNDecoder(
            ca_dgn_dim=ca_dgn_dim,
            expansion=decoder_expansion,
            dropout=decoder_dropout,
        )

        self.graph_builder = GraphBuilder(
            hidden_dim=ca_dgn_dim,
        )

    @classmethod
    def load(
            cls,
            path: str | Path,
            model_id: str,
            map_location: Optional[str | torch.device] = None,
    ):
        path = Path(path)
        with open(path / f"{self.SAVE_LOAD_PREFIX}_{model_id}_config.json") as f:
            config = json.load(f)
        model = cls(**config)
        state = torch.load(path / f"{self.SAVE_LOAD_PREFIX}_{model_id}_weights.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        return model

    def configure_stage1(self) -> None:
        self.encoder.requires_grad_(True)
        self.decoder.requires_grad_(True)
        self.graph_builder.requires_grad_(True)

    def configure_stage2(self) -> None:
        self.encoder.requires_grad_(False)
        self.decoder.requires_grad_(False)
        self.graph_builder.requires_grad_(False)

    @property
    def __config__(self) -> dict[str, Any]:
        return self._config

