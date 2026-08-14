import pandas as pd
from pathlib import Path
from typing import Union
import gc
import torch


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
