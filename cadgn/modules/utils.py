import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import gc
import torch
from collections import defaultdict
import random
import torch.nn as nn


def flush_gpu() -> None:
    """
    Force garbage collection and clear PyTorch's CUDA memory cache.
    Call once at process startup to clear fragmentation from prior
    sessions. Safe to call on CPU-only machines; no-op if CUDA
    is unavailable.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"GPU: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved")


def save_history(
        history: dict[str, list[float]],
        path: Union[Path, str]
):
    """
    Saves the training history to a csv file.
    An epoch column is prepended for clarity.
    :param history: dict[metric_name, list[per_epoch_value]]
    :param path:  File path to write the CSV to. Parent directories created if absent.
    :return: The dataframe.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(history)
    df.insert(0, "epoch", range(len(df)))

    df.to_csv(path, index=False)
    return df


@dataclass
class Stage2Intermediates:
    """
    Stores detached versions of all intermediate tensors for evaluation in Stage2 (for effectiveness of model architecture)
    """
    H: torch.Tensor  # EncoderHead output
    Z: torch.Tensor  # CADGNEncoder output
    int_Z: torch.Tensor  # Pre-Projector output (to LLM)
    Z_approx: torch.Tensor  # Post-Projector output (to CADGNDecoder & GraphBuilder)

    @classmethod
    def from_step(
            cls,
            H: torch.Tensor,
            Z: torch.Tensor,
            int_Z: torch.Tensor,
            Z_approx: torch.Tensor,
    ) -> "Stage2Intermediates":
        """

        :param H: EncoderHead output
        :param Z: CADGNEncoder output
        :param int_Z: Pre-Projector output (to LLM)
        :param Z_approx: Post-Projector output (to CADGNDecoder and GraphBuilder)
        :return: Stage2Intermediates object (all tensors detached)
        """
        return cls(
            H = H.detach(),
            Z = Z.detach(),
            int_Z = int_Z.detach(),
            Z_approx = Z_approx.detach(),
        )


def is_large_model(model_id: str) -> bool:
    return "-8B" in model_id


def get_device_of(module: nn.Module) -> torch.device:
    """Infer current device from first parameter or buffer. Falls back to CPU."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        pass
    try:
        return next(module.buffers()).device
    except StopIteration:
        return torch.device("cpu")