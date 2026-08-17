from dataclasses import dataclass


@dataclass
class BaselineSampleResult:
    sample_id: int
    rung: int
    label: str
    prediction: str
    yn_correct: float
    confidence: float
    yn_coverage: float
    gate_pred: int
    gate_correct: float
    gate_logits: list[float]  # stored as plain list; JSON serialisable

