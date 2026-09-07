import torch
import torch.nn as nn
from typing import Union
from cadgn import get_device_of


class CudaDevice:
    """
    Context manager that moves nn.Module(s) to a target device on entry, and restores them to their original device on exit.
    """
    def __init__(self, *modules: nn.Module, device: Union[str, torch.device] = "cuda"):
        self.modules = list(modules)
        self.device = device
        self._original_devices: list[torch.device] = []

    def __enter__(self) -> "CudaDevice":
        for module in self.modules:
            self._original_devices.append(get_device_of(module))
            module.to(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module, original in zip(self.modules, self._original_devices):
            module.to(original)
        return False
