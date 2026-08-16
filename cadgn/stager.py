from __future__ import annotations
from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from pathlib import Path
import json
from typing import Any, Iterable, Optional


class Staged(ABC):
    SAVE_LOAD_PREFIX = "Staged"

    @abstractmethod
    def configure_stage1(self) -> None:
        ...

    @abstractmethod
    def configure_stage2(self) -> None:
        ...

    def _save_extra(self, path: Path, model_id: str, *args, **kwargs):
        pass

    def save(self, path: str | Path, model_id: str, *args, **kwargs) -> None:
        """
        Save hyperparameters and weights to a directory.
        Directory layout (minimum):
            {path}/
                {SAVE_LOAD_PREFIX}_{model_id}_config.json ; self._config, for module reconstruction
                {SAVE_LOAD_PREFIX}_{model_id}_weights.pt  ; self.state_dict()
        Subclasses needing additional artifacts should override `_save_extra` rather than this method.
        :param path: Target directory. Created (including parents) if absent.
        :param model_id: Model ID (to disambiguate instances)
        :return:
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / f"{self.SAVE_LOAD_PREFIX}_{model_id}_config.json", "w") as f:
            json.dump(self.__config__, f, indent=2)
        self._save_extra(path, model_id, *args, **kwargs)
        torch.save(self.state_dict(), path / f"{self.SAVE_LOAD_PREFIX}_{model_id}_weights.pt")

    def _save_extra(self, path, model_id, *args, **kwargs) -> None:
        """
        Hook for Subclass Specific Save Artifacts
        Called by save() after config.json is written, but before weights.pt
        Default is a no-op.
        :param path: Target directory. Created (including parents) if absent.
        :param model_id: Model ID (to disambiguate instances)
        :param args: Any positional args required
        :param kwargs: Any keyword positional args required
        :return:
        """
        pass

    @classmethod
    @abstractmethod
    def load(
            cls,
            path: str | Path,
            model_id: str,
            map_location: Optional[str | torch.device] = None,
    ):
        """
        Reconstruct this component from a directory produced by save()
        :param path:  Directory containing config.json and weights.pt (and any other files required)
        :param model_id:
        :param map_location: Passed to torch.load for device remapping.
        :return:
        """
        ...

    @property
    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """
        Yield only parameters that currently require gradients.
        Usage:
            optimizer = torch.optim.AdamW(family.trainable_parameters(), lr=1e-4)
        :return:
        """
        if hasattr(self, "parameters"):
            return (p for p in self.parameters() if p.requires_grad)
        return tuple()

    def __getitem__(self, item: str):
        return self.__dict__.get(item, self.__config__.get(item, None))

    @property
    @abstractmethod
    def __config__(self) -> dict[str, Any]:
        ...

