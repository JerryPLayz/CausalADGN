from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Note: ensure that the output is properly softmax'd. Output classes: 0-2
class GateClassifier(nn.Module):
    """
    Purpose is to act as a diagnostic tool of the LLM's prefill state: can the LLM appropriately determine the task it is being asked to do?
    (Associational, Interventional, Counterfactual)
    """
    def __init__(self,
                 llm_dim: int,
                 scale: int = 8,
                 dropout: float = 0.1,
                 ):
        super(GateClassifier, self).__init__()
        self.llm_dim = llm_dim
        self.dropout_pc = dropout
        self.scale = scale

        # build layers geometrically

        dims = [llm_dim]
        while True:
            next_dim = dims[-1] // scale
            if next_dim <= 3:
                break
            dims.append(next_dim)
        # i.e. [4096, 512, 64]

        # assemble layers
        blocks: list[nn.Module] = []
        for d_in, d_out in zip(dims, dims[1:]):
            blocks += [
                nn.Linear(d_in, d_out),
                nn.GELU(),
                nn.Dropout(p=self.dropout_pc),
            ]

        # final projection to output dimensionality (no activation, 3)
        blocks.append(nn.Linear(dims[-1], 3))
        self.layers = nn.Sequential(*blocks)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
