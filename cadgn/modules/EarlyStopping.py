class EarlyStopping:
    """
    utility to prevent wasted compute. If the model does not improve after `patience` epochs by at least `min_delta`, then
    end training early.
    """
    def __init__(
            self,
            patience: int = 10,
            min_delta: float = 1e-4,
            mode: str ="min"
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._counter: int = 0
        self.best_epoch: int = 0

    def step(self, metric: float, epoch: int) -> bool:
        """
        Returns True if training should stop.
        """
        improved = (
            metric < self._best - self.min_delta
            if self.mode == "min"
            else metric > self._best + self.min_delta
        )
        if improved:
            self._best = metric
            self._counter = 0
            self.best_epoch = epoch
        else:
            self._counter += 1

        return self._counter >= self.patience

    @property
    def best(self) -> float:
        return self._best

