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

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "rung": self.rung,
            "label": self.label,
            "prediction": self.prediction,
            "yn_correct": self.yn_correct,
            "confidence": self.confidence,
            "yn_coverage": self.yn_coverage,
            "gate_pred": self.gate_pred,
            "gate_correct": self.gate_correct,
            # leave out gate_logits of the dict - this is for saving only.
        }
