import json
from . import BaselineSampleResult
from pathlib import Path
from typing import Union, List


def save_sample_results(
        results: List[BaselineSampleResult],
        path: Union[str, Path],
        variant: str,
        epoch: int
) -> None:
    """
    Save per-sample evaluation results to JSON for statistical testing.
    File is keyed by variant name and epoch, so results from all ablation variants can be loaded together for comparison.
    :param results: Per-sample records from _eval()
    :param path: Output directory
    :param variant: Ablation variant name (e.g. baseline, cadgn, adgn, gat etc.)
    :param epoch:  Epoch number (or sub-variant number, given there may need to be a distinction for graph first vs graph last and w_mmd (to test the significance of that)
    :return:
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "sample_id": r.sample_id,
            "rung": r.rung,
            "label": r.label,
            "prediction": r.prediction,
            "yn_correct": r.yn_correct,
            "confidence": r.confidence,
            "yn_coverage": r.yn_coverage,
            "gate_pred": r.gate_pred,
            "gate_correct": r.gate_correct,
            "gate_logits": r.gate_logits,
        }
        for r in results
    ]

    fname = path / f"{variant}_{epoch:04d}_per_sample.json"
    with open(fname, "w") as f:
        json.dump(records, f, indent=4)


