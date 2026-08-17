import torch
import torch.nn as nn
import torch.functional as F
from typing import Any, Callable, Dict, Optional, Union


class Projector(nn.Module):
    def __init__(
            self,
            d_from: int,
            d_to: int,
            expansion: int = 2,
            dropout: float = 0.1,
            output_dtype: Optional[torch.dtype] = None
    ) -> None:
        super().__init__()
        self.d_from = d_from
        self.d_to = d_to
        self.expansion = expansion
        self.output_dtype = output_dtype

        d_hidden = max(d_from, d_to) * expansion

        self.proj = nn.Sequential(
            nn.Linear(d_from, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_to),
            nn.LayerNorm(d_to),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        Z = Z.to(next(self.parameters()).dtype)
        out = self.proj(Z)
        if self.output_dtype is not None:
            out = out.to(self.output_dtype)
        return out



