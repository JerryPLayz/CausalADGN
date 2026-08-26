# This file sets out the methods used to statistically assess model performance.

"""
Notes:

    1. McNemar's Test (for binary yes/no classification)
    Tests whether two variants disagree in the same way on the individual samples.
    H0: The two variants have equivalent error patterns
    Auto-selects exact (binomial) when b+c < 25, chi-squared otherwise

    2. Stuart-Maxwell Test (3-class GateClassifier)
    A generalization of McNemar for KxK paired contingency tables.
    H0: Marginal homogeneity, the two variants have equivalent rung prediciton distributions.
    Implemented by statsmodel SquareTable.symmetry()

    3. Almost Stochastic Order (for continuous metrics)
     Tests whether variant A stochastically dominates variant B across the full score distribution.
     Returns epsilon_min, an upper bound on the violation of stochastic dominance.
     H0: epsilon_min >= tau (no dominance)
     Lower epsilon_mind -> stronger dominance of A over B.
     - If epsilon_min < 0.5 with alpha=0.05: A is declared superior.
     Implemented via deepsig.aso

To ensure we don't over-estimate support, we employ the Holm-Bonferroni correction to control for the Family-Wise Error Rate.


"""


from __future__ import annotations

from collections import defaultdict

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import chi2
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar, SquareTable
from statsmodels.stats.multitest import multipletests
from typing import Optional, Union, List

try:
    from deepsig import aso as deepsig_aso
    _DEEPSIG_AVAILABLE = True
except ImportError:
    _DEEPSIG_AVAILABLE = False

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

mp.set_start_method("spawn", force=True)


# Just something to try to speed this up.... Its painfully slow...
_worker_variants: list[VariantResults] | None = None
_worker_variants_aso: list[VariantResults] | None = None


def _init_worker(variants: list[VariantResults]) -> None:
    global _worker_variants
    _worker_variants = variants


def _init_worker_aso(variants: list[VariantResults]) -> None:
    global _worker_variants_aso
    _worker_variants_aso = variants


def _stuart_maxwell_worker(args: tuple[int, int]) -> tuple[int, int, float, float, str, int, int, str]:
    i, j = args
    stat, pval, direction, direction_int, n, notes = _stuart_maxwell_pair(
        _worker_variants[i],
        _worker_variants[j],
    )
    return i, j, stat, pval, direction, direction_int, n, notes


def _aso_worker(
        args: tuple[int, int, str, float, int]
) -> tuple[int, int, str, float, str, int, int, str]:
    i, j, metric, confidence, seed = args
    eps, direction, direction_int, n, notes = _aso_pair(
        _worker_variants_aso[i],
        _worker_variants_aso[j],
        metric=metric,
        confidence=confidence,
        seed=seed,
        num_jobs=1,  # disable internal parallelism, outer pool handles it
    )
    return i, j, metric, eps, direction, direction_int, n, notes


@dataclass
class VariantResults:
    """
    Per-sample evaluation records for one ablation variant, aligned by sample_id.
    """
    name: str
    records: list[dict]
    sample_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.sample_ids = [r["sample_id"] for r in self.records]

    def get_scores(self, fld: str):
        """
        Extract a numerical field, ordered by
        :param fld: a key from BaselineSampleResult
        :return: (n_samples, ) float array.
        """
        return np.array([r[fld] for r in self.records])

    def get_binary_correct(self) -> np.ndarray:
        """
        yn_correct as a strict binary; 1=correct.
        Abstain results are incorrect for the purposes of the McNemar test.
        :return:
        """
        return (np.array([r["yn_correct"] for r in self.records]) == 1.0).astype(int)


@dataclass
class PairwiseResult:
    """
    Result of a single pairwise statistical test between two variants (ablations).
    Attributes:
        variant_a: Name of variant A (e.g. 'cadgn')
        variant_b: Name of variant B (e.g. 'adgn')
        test: Test name ("mcnemar", "stuart_maxwell", "aso")
        metric: Metric being tested (e.g. "yn_correct")
        statistic: Test statistic (e.g. chi2 for McNemar/Stuart-Maxwell, epsilon_min for ASO)
        p_value: p-value (None for ASO)
        p_value_corr: Holm-Bonferroni corrected p-value (None for ASO)
        reject_h0: whether H0 is rejected at corrected alpha
        direction: "A > B" (1), "B > A" (-1), or "no difference" (0)
        direction_int: takes on an integer value, representing the cases shown in `direction`
        n_samples: Number of aligned samples used.
        notes: Any warnings or caveats.
    """
    variant_a: str
    variant_b: str
    test: str
    metric: str
    statistic: float
    p_value: Optional[float]
    p_value_corr: Optional[float]
    reject_h0: bool
    direction: str
    direction_int: int
    n_samples: int
    notes: str


@dataclass
class AnalysisReport:
    """
    Full pairwise analysis results across all variants and metrics.
    Attributes:
        results: All pairwise test results
        alpha: Family-wise error rate used.
        n_tests_mcnemar: number of mcnemar tests run
        n_tests_stuart_maxwell: number of stuart-maxwell tests run
        n_tests_aso: Number of ASO tests run per continuous metric.
    """
    results: list[PairwiseResult]
    alpha: float
    n_tests_mcnemar: int
    n_tests_stuart_maxwell: int
    n_tests_aso: int

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "variant_a": r.variant_a,
                "variant_b": r.variant_b,
                "test": r.test,
                "metric": r.metric,
                "statistic": r.statistic,
                "p_value": r.p_value,
                "p_value_corr": r.p_value_corr,
                "reject_h0": r.reject_h0,
                "direction": r.direction,
                "direction_int": r.direction_int,
                "n_samples": r.n_samples,
                "notes": r.notes,
            }
            for r in self.results
        ])

    def summary(self) -> str:
        """
        Prints a human-readable summary of the results.
        Significant findings only (e.g. where H0 is rejected after correction)
        """
        df = self.to_dataframe()
        sig = df[df["reject_h0"] == True]

        lines = [
            f"Analysis Report (alpha={self.alpha}, Holm-Bonferroni corrected)",
            f"\tMcNemar tests:         {self.n_tests_mcnemar}",
            f"\tStuart-Maxwell tests:  {self.n_tests_stuart_maxwell}",
            f"\tASO tests:             {self.n_tests_aso} (per continuous metric)",
            f"\tSignificant Results:   {len(sig)}",
            "",
        ]

        if sig.empty:
            lines.append(f"\t>>No significant results after correction.")
        else:
            for _, row in sig.iterrows():
                lines.append(
                    f" [{row['test']}] {row['variant_a']} vs {row['variant_b']} "
                    f"| {row['metric']} | stat={row['statistic']:.4f} "
                    f"| p_corr={row['p_value_corr']:.4f} "
                    f"| {row['direction']}"
                )
        return "\n".join(lines)

    def save(self, path: str | Path, model_id: str) -> None:
        """
        Saves full results to CSV and summary to text
        :param path:
        :return:
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.to_dataframe().to_csv(path / f"{model_id}_statistics.csv", index=False)

        with open(path / f"{model_id}_statistics_summary.txt", "w") as f:
            f.write(self.summary())


def load_variant_results(
        files: dict[tuple[str, str, int], Union[str, Path]],
        epoch: int,
        variant_names: list[str],
) -> list[VariantResults]:
    """
    Load per-sample evaluation results for a set of ablation variants.
    :param files: dict with key (ablation, modelid, mmd_int), and value of the path to the instance with that configuration.
    :param epoch:
    :param variant_names:
    :return:
    """
    loaded = []
    for k, v in files.items():
        (abl, modelid, mmd_int) = k
        p = Path(v)
        name = f"{abl}_mmd-{mmd_int}_{modelid}"
        if not p.exists():
            raise FileNotFoundError(f"No file found at the path provided: {p}.")

        with open(p, "r") as f:
            records = json.load(f)
        loaded.append(VariantResults(name=name, records=records))
        print(f"\tLoaded {len(records)} samples for variant")

    all_id_sets = [set(v.sample_ids) for v in loaded]
    common_ids = set.intersection(*all_id_sets)
    if len(common_ids) < max(len(s) for s in all_id_sets):
        print(
            f"\nWARNING: Variants do not share identical sample sets.\n"
            f" Common Samples: {len(common_ids)}\n"
            f" Max Variant Samples: {max(len(s) for s in all_id_sets)}\n"
            f" Pairwise tests will use only common samples."
        )

    return loaded

def _align_pair(
        va: VariantResults,
        vb: VariantResults,
) -> tuple[list[dict], list[dict]]:
    """
    Align two VariantResults by sample_id, returning matched record pairs.
    Only samples present in both variants are included.
    :param va: VariantResult object 1
    :param vb: VariantResult object 2
    :return: (records_a, records_b), parallel lists, same length
    """
    b_map = {r["sample_id"]: r for r in vb.records}
    matched_a, matched_b = [], []

    for ra in va.records:
        sid = ra["sample_id"]
        if sid in b_map:
            matched_a.append(ra)
            matched_b.append(b_map[sid])
    return matched_a, matched_b


def _mcnemar_pair(
        va: VariantResults,
        vb: VariantResults,
) -> tuple[float, float, str, int, int, str]:
    """
    McNemar's test for binary yes/no classification between two variants.
    Contingency table in format:
                B correct   B incorrect
    A correct       n00           n01 (b)
    A incorrect     n10 (c)       n11
    Test statistic: chi2 = (b - c)^2 / (b+c)
    (with continuity correction when b+c >= 25)
    Exact binomial when b+c < 25 (as recommended by Dietterich, 1998)
    :param va:
    :param vb:
    :return: (statistic, p_value, direction, direction_int, n_samples, notes)
    """
    recs_a, recs_b = _align_pair(va, vb)
    n_samples = len(recs_a)

    if n_samples == 0:
        return float("nan"), float("nan"), "no shared samples", 0, 0, "no shared samples"

    correct_a = np.array([1 if r["yn_correct"] == 1.0 else 0 for r in recs_a])
    correct_b = np.array([1 if r["yn_correct"] == 1.0 else 0 for r in recs_b])

    # Build 2x2 contingency table
    n00 = int(((correct_a == 1) & (correct_b == 1)).sum())
    n01 = int(((correct_a == 1) & (correct_b == 0)).sum())  # b
    n10 = int(((correct_a == 0) & (correct_b == 1)).sum())  # c
    n11 = int(((correct_a == 0) & (correct_b == 0)).sum())

    table = np.array([[n00, n01], [n10, n11]])
    bc = n01 + n10

    notes = ""
    if bc == 0:
        # Perfect agreement, test not meaningful
        return 0.0, 1.0, "no difference", 0, n_samples, "b+c=0: perfect agreement"

    # Auto-Select Exact vs Chi-Squared
    use_exact = bc < 25
    if use_exact:
        notes = f"exact test (b+c={bc}<25)"

    result = sm_mcnemar(table, exact=use_exact, correction= not use_exact)
    #print(result)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)

    # Direction
    acc_a = correct_a.mean()
    acc_b = correct_b.mean()

    if acc_a > acc_b:
        direction = f"A > B ({acc_a:.3f} > {acc_b:.3f})"
        direction_int = 1
    elif acc_b > acc_a:
        direction = f"B > A ({acc_b:.3f} > {acc_a:.3f})"
        direction_int = -1
    else:
        direction = "no difference"
        direction_int = 0

    return statistic, p_value, direction, direction_int, n_samples, notes

def _stuart_maxwell_pair(
        va: VariantResults,
        vb: VariantResults,
        n_classes:int = 3
) -> tuple[float, float, str, int, int, str]:
    """
    Stuart-Maxwell test for 3-class GateClassifier predictions.
    Builds a KxK contingency table where entry (i,j) is the number of samples where A predicted class i and B predicted class j.
    H0: Marginal homogeneity; the distribution of the rung predictions is equivalent betwen the two variants across all samples.

    A significant result means the variants make systematically different rung classification errors, not necessarily that one is more accurate.
    Consult Stuart, A. (1955); Maxwell, A.E. (1970)

    :param va:
    :param vb:
    :param n_classes:
    :return: (statistic, p_value, direction, direction_int, n_samples, notes)
    """
    recs_a, recs_b = _align_pair(va, vb)
    n_samples = len(recs_a)

    if n_samples == 0:
        return float("nan"), float("nan"), "no shared samples", 0, 0, "no shared samples"

    preds_a = np.array([r["gate_pred"] for r in recs_a])
    preds_b = np.array([r["gate_pred"] for r in recs_b])

    # Build K×K contingency table
    table = np.zeros((n_classes, n_classes), dtype=int)
    for pa, pb in zip(preds_a, preds_b):
        if 0 <= pa < n_classes and 0 <= pb < n_classes:
            table[pa, pb] += 1

    # Check for degenerate table (all predictions identical)
    if table.sum() == 0:
        return 0.0, 1.0, "no difference", 0, n_samples, "empty contingency table"

    sq = SquareTable(table)
    result = sq.symmetry()  # Stuart-Maxwell test

    statistic = float(result.statistic)
    p_value = float(result.pvalue)

    acc_a = np.array([r["gate_correct"] for r in recs_a]).mean()
    acc_b = np.array([r["gate_correct"] for r in recs_b]).mean()

    if acc_a > acc_b:
        direction = f"A > B (gate_acc: {acc_a:.3f} > {acc_b:.3f})"
        direction_int = 1
    elif acc_b > acc_a:
        direction = f"B > A (gate_acc: {acc_b:.3f} > {acc_a:.3f})"
        direction_int = -1
    else:
        direction = "no difference"
        direction_int = 0

    notes = f"df={n_classes * (n_classes - 1) // 2}, n={n_samples}"
    return statistic, p_value, direction, direction_int, n_samples, notes


def _aso_pair(
        va: VariantResults,
        vb: VariantResults,
        metric: str,
        confidence: float = 0.95,
        seed: int = 42,
        num_jobs: int = 1,
) -> tuple[float, str, int, int, str]:
    """
    Almost Stochastic Order test for a continuous metric.
    ASO computes epsilon_mind, an upper bound on the violation of stochastic dominance of A over B.

    Interpretation:
        epsilon_min < 0.5: A stochastically dominates B
        epsilon_max ~ 0.5: No dominance (distributions are equivalent)
        epsilon_min > 0.5: B stochastically dominates A
    Rejection criteria:
        Reject H0 (epsilon_min >= tau) when epsilon_min < tau.

    Consult:
        del barrio et al. (2017) "Optimal transport and robust statistics"
        Dror et al. (2019) "Deep Dominance - How to properly compare DNN models"
        Ulmer et al. (2022) "Deep Significance"
    :param va:
    :param vb:
    :param metric:
    :param confidence:
    :param seed:
    :param num_jobs:
    :return: (epsilon_min, direction, direction_int, n_samples, notes)
    """

    if not _DEEPSIG_AVAILABLE:
        return float("nan"), "deepsig not installed", 0, 0, "pip install deepsig"

    recs_a, recs_b = _align_pair(va, vb)
    n_samples = len(recs_a)

    if n_samples == 0:
        return float("nan"), "no shared samples", 0, 0, "no shared samples"

    scores_a = np.array([r[metric] for r in recs_a], dtype=float)
    scores_b = np.array([r[metric] for r in recs_b], dtype=float)

    # Sanitise: drop NaN pairs
    valid = ~(np.isnan(scores_a) | np.isnan(scores_b))
    scores_a = scores_a[valid]
    scores_b = scores_b[valid]
    n_valid = int(valid.sum())

    if n_valid == 0:
        return float("nan"), "no valid samples", 0, 0, "all NaN"

    # epsilon_min for A > B: lower = A more dominant
    eps_min = float(deepsig_aso(
        scores_a,
        scores_b,
        confidence_level=confidence,
        num_jobs=num_jobs,
        seed=seed,
    ))

    mean_a = float(scores_a.mean())
    mean_b = float(scores_b.mean())

    if eps_min < 0.5:
        direction = f"A > B (mean: {mean_a:.4f} > {mean_b:.4f}, eps={eps_min:.4f})"
        direction_int = 1
    else:
        direction = f"B >= A (mean: {mean_b:.4f} > {mean_a:.4f}, eps={eps_min:.4f})"
        direction_int = -1

    notes = (
        f"confidence={confidence}, n_valid={n_valid}/{n_samples}. "
        f"Report epsilon_min and confidence level. "
        f"Reject H0 (no dominance) when eps_min < 0.5."
    )
    return eps_min, direction, direction_int, n_samples, notes


def _apply_holm_bonferroni(
        raw_results: list[PairwiseResult],
        alpha: float = 0.05,
) -> list[PairwiseResult]:
    """
    Apply Holm-Bonferroni correction to a set of pairwise resutls sharing the same type and metric.
    Applied separately per (test_type, metric) group.

    :param raw_results: PairwiseResult list with p_value set, p_value_corr=None
    :param alpha: Family-wise error rate (e.g. 0.05)
    :return: Same list with p_value_corr and reject_h0 filled in.
    """
    # Filter to results with valid p-values (ASO results have None)
    testable = [r for r in raw_results if r.p_value is not None
                and not np.isnan(r.p_value)]

    if not testable:
        return raw_results

    p_values = np.array([r.p_value for r in testable])

    reject, p_corrected, _, _ = multipletests(
        p_values,
        alpha=alpha,
        method="holm",
    )

    # Write corrected values back
    for i, result in enumerate(testable):
        result.p_value_corr = float(p_corrected[i])
        result.reject_h0 = bool(reject[i])

    return raw_results


def run_analysis(
        variants: list[VariantResults],
        alpha: float = 0.05,
        aso_confidence: float = 0.95,
        aso_tau: float = 0.5,
        aso_seed: int = 42,
        num_cpus: int = 15,
        continuous_metrics: Optional[List] = None
) -> AnalysisReport:
    """
    Run full pairwise statistical analysis across all ablation variants.
    Performs three test types, each with Holm-Bonferroni correction applied within that test type:
        1. McNemar; binary yn_correct across all pairs
        2. Stuart-Maxwell; 3-class gate_pred across all pairs
        3. ASO; each continuous metric across all pairs.
    Interpretation:
        1. McNemar / Stuart-Maxwell; reject_h0=True means the variants differ in their error patters significantly (direction informs which model is better)
        2. ASO; reject_h0=True means epsilon_min < aso_tau: A stochastically dominates B.

    :param variants: list of VariantResults from load_variant_results(
    :param alpha: Family-wise error rate (default. 0.05)
    :param aso_confidence: Bootstrap confidence level for ASO (default 0.95)
    :param aso_tau: ASO rejection threshold (default 0.5). Set lower (e.g. 0.2) for stricter dominance criterion
    :param aso_seed: Random seed for ASO bootstrap (default 42)
    :param aso_num_jobs: Parallel jobs for ASO (default 1)
    :param continuous_metrics: Metrics to test with ASO. Default ["confidence", "yn_coverage"]
    :return: AnalysisReport with all pairwise results and corrections applied.
    """
    if continuous_metrics is None:
        continuous_metrics = ["confidence", "yn_coverage"]

    pairs = list(combinations(range(len(variants)), 2))
    n_pairs = len(pairs)

    # 1. McNemar (binary yn_correct)
    print(f"\t>>> 1 | Starting McNemar Pairwise Testing...")
    mcnemar_results: list[PairwiseResult] = []
    for idx, (i, j) in enumerate(pairs):
        if idx % 100 == 0:
            print(f"\t\t{idx}/{n_pairs}...")
        va, vb = variants[i], variants[j]
        stat, pval, direction, direction_int, n, notes = _mcnemar_pair(va, vb)
        mcnemar_results.append(PairwiseResult(
            variant_a=va.name, variant_b=vb.name,
            test="mcnemar",
            metric="yn_correct",
            statistic=stat,
            p_value=pval,
            p_value_corr=None,
            reject_h0=False,
            direction=direction,
            direction_int=direction_int,
            n_samples=n,
            notes=notes,
        ))
    _apply_holm_bonferroni(mcnemar_results, alpha)


    # 2. Stuart-Maxwell (3-class gate_pred)
    print(f"\t>>> 2 | Starting Stuart-Maxwell Pairwise Testing... ({n_pairs} pairs)")
    sm_results: list[PairwiseResult] = []
    n_workers = min(num_cpus, mp.cpu_count())
    completed = 0

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(variants,)
    ) as executor:
        futures = {
            executor.submit(_stuart_maxwell_worker, (i, j)): (i, j)
            for i, j in pairs
        }
        for future in as_completed(futures):
            i, j, stat, pval, direction, direction_int, n, notes = future.result()
            va, vb = variants[i], variants[j]
            sm_results.append(PairwiseResult(
                variant_a=va.name,
                variant_b=vb.name,
                test="stuart_maxwell",
                metric="gate_pred",
                statistic=stat,
                p_value=pval,
                p_value_corr=None,
                reject_h0=False,
                direction=direction,
                direction_int=direction_int,
                n_samples=n,
                notes=notes,
            ))
            completed += 1
            if completed % 100 == 0:
                print(f"\t\t{completed}/{n_pairs}...")
        # EOL
    # EOExec
    _apply_holm_bonferroni(sm_results, alpha)

    # 3. ASO (continuous metrics)
    aso_results: list[PairwiseResult] = []
    if not _DEEPSIG_AVAILABLE:
        print("\nWARNING: `deepsig` not installed, ASO tests skipped.\n")
    else:
        print(f"\t>>> 3 | Starting ASO Pairwise Testing...")
        # Build all tasks upfront: one per (pair, metric) combination
        all_tasks: list[tuple[int, int, str, float, int]] = [
            (i, j, metric, aso_confidence, aso_seed)
            for metric in continuous_metrics
            for i, j in pairs
        ]
        n_tasks = len(all_tasks)
        completed = 0

        # Accumulate per-metric for Holm-Bonferroni (applied per metric)
        metric_buckets: defaultdict[str, list[PairwiseResult]] = defaultdict(list)

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker_aso,
            initargs=(variants,)
        ) as executor:
            futures = {
                executor.submit(_aso_worker, task): task
                for task in all_tasks
            }
            for future in as_completed(futures):
                i, j, metric, eps, direction, direction_int, n, notes = future.result()
                va, vb = variants[i], variants[j]
                metric_buckets[metric].append(PairwiseResult(
                    variant_a=va.name,
                    variant_b=vb.name,
                    test="aso",
                    metric=metric,
                    statistic=eps,
                    p_value=None,
                    p_value_corr=None,
                    reject_h0=(
                            not np.isnan(eps) and eps < aso_tau
                    ),
                    direction=direction,
                    direction_int=direction_int,
                    n_samples=n,
                    notes=notes,
                ))
                completed += 1
                if completed % 100 == 0:
                    print(f"\t\t{completed}/{n_tasks}...")
        # Flatten in metric order (preserves original structure)
        for metric in continuous_metrics:
            aso_results.extend(metric_buckets[metric])
    print(f"\t >>> Process Complete!")
    all_results = mcnemar_results + sm_results + aso_results

    return AnalysisReport(
        results=all_results,
        alpha=alpha,
        n_tests_mcnemar=len(mcnemar_results),
        n_tests_stuart_maxwell=len(sm_results),
        n_tests_aso=len(aso_results),
    )
