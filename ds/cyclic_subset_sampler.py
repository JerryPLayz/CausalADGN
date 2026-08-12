from torch.utils.data import Sampler
import torch


class CyclicSubsetSampler(Sampler):
    """
    Cycles through dataset indices in shuffled order, yielding exactly `max_steps` indices
    per epoch. Guarantees that every sample is seen at least once per `ceil(dataset_size / max_steps)` epochs.

    When the shuffled index is exhausted mid-epoch, it reshuffles and continues from the beginning.
    """

    def __init__(
            self,
            dataset_size: int,
            max_steps: int,
            seed: int = 42
    ):
        super().__init__()
        self.dataset_size = dataset_size
        self.max_steps = max_steps
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

        # Initialize with a full shuffled pass
        self._indices: list[int] = torch.randperm(
            dataset_size, generator=self._rng
        ).tolist()
        self._pos: int = 0

    def _reshuffle(self) -> None:
        self._indices: list[int] = torch.randperm(
            self.dataset_size, generator=self._rng
        ).tolist()
        self._pos = 0

    def __iter__(self):
        indices: list[int] = []
        remaining = self.max_steps
        while remaining > 0:
            available = self.dataset_size - self._pos
            if available <= remaining:
                # Take what's left and then reshuffle
                indices.extend(self._indices[self._pos:])
                remaining -= available
                self._reshuffle()
            else:
                indices.extend(self._indices[self._pos:self._pos + remaining])
                self._pos += remaining
                remaining = 0
        return iter(indices)

    def __len__(self) -> int:
        return self.max_steps
