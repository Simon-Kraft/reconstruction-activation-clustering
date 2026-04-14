"""
evaluate.py — Detection evaluation against ground truth poison labels.

This is where your core experimental results come from.

Since you constructed the poisoned dataset yourself, you have perfect
ground truth for every training sample (poisoned or clean). After AC
produces per-sample predictions, this module compares them against
ground truth and computes:

  Accuracy:         (TP + TN) / total
  F1 score:         2TP / (2TP + FP + FN)
  Confusion matrix: [[TN, FP], [FN, TP]]

These are computed per class and aggregated, matching the format of
Table 1 in Chen et al. (2018) so you can directly compare your results.

Why both accuracy AND F1:
    With a 15% poison rate, 85% of samples are clean. A naive detector
    that calls everything clean would score 85% accuracy — misleadingly
    high. F1 penalises heavily for missing poisoned samples, which is
    exactly the failure mode that matters. The paper reports both;
    you should too.
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

from activation_clustering.extractor import ExtractionResult
from activation_clustering.analyzer  import AnalysisResult


# ---------------------------------------------------------------------------
# Per-class result
# ---------------------------------------------------------------------------

@dataclass
class ClassEvalResult:
    """
    Evaluation metrics for a single class.

    Attributes:
        cls:        class label
        accuracy:   fraction of samples correctly classified as clean/poisoned
        f1:         F1 score for the poisoned class (positive label = poisoned)
        precision:  TP / (TP + FP)
        recall:     TP / (TP + FN)  — fraction of poisoned samples caught
        tp:         true positives  (predicted poisoned, actually poisoned)
        tn:         true negatives  (predicted clean,   actually clean)
        fp:         false positives (predicted poisoned, actually clean)
        fn:         false negatives (predicted clean,   actually poisoned)
        n_total:    total samples in this class
        n_poison:   ground truth poisoned count
        is_target:  True if this is the target class (should be poisoned)
    """
    cls:       int
    accuracy:  float
    f1:        float
    precision: float
    recall:    float
    tp:        int
    tn:        int
    fp:        int
    fn:        int
    n_total:   int
    n_poison:  int
    is_target: bool


# ---------------------------------------------------------------------------
# Full evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    Full evaluation across all classes.

    Attributes:
        per_class:       dict mapping class label → ClassEvalResult
        overall_accuracy: accuracy across all samples in all classes
        overall_f1:       F1 across all samples in all classes
        target_class:     which class was poisoned
    """
    per_class:        dict[int, ClassEvalResult]
    overall_accuracy: float
    overall_f1:       float
    target_class:     int

    def print_table(self) -> None:
        """Print results in the style of Table 1 from Chen et al. (2018)."""
        print("\n" + "=" * 70)
        print("  Detection Results  (cf. Chen et al. 2018, Table 1)")
        print("=" * 70)

        # Header
        print(
            f"  {'class':>5}  {'accuracy':>9}  {'f1':>8}  "
            f"{'precision':>9}  {'recall':>7}  "
            f"{'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>6}"
        )
        print("  " + "-" * 67)

        # Per-class rows
        for cls in sorted(self.per_class.keys()):
            r      = self.per_class[cls]
            marker = " ←" if r.is_target else ""
            print(
                f"  {cls:>5}  "
                f"{r.accuracy:>9.2%}  "
                f"{r.f1:>8.4f}  "
                f"{r.precision:>9.4f}  "
                f"{r.recall:>7.4f}  "
                f"{r.tp:>4}  {r.fp:>4}  {r.fn:>4}  {r.tn:>6}"
                f"{marker}"
            )

        # Overall row
        print("  " + "-" * 67)
        print(
            f"  {'TOTAL':>5}  "
            f"{self.overall_accuracy:>9.2%}  "
            f"{self.overall_f1:>8.4f}"
        )
        print("=" * 70)
        print(f"  ← = target class (class {self.target_class})\n")

    def to_dataframe(self) -> pd.DataFrame:
        """Return results as a DataFrame for saving/plotting."""
        rows = []
        for cls, r in self.per_class.items():
            rows.append({
                "class":     cls,
                "accuracy":  r.accuracy,
                "f1":        r.f1,
                "precision": r.precision,
                "recall":    r.recall,
                "tp":        r.tp,
                "tn":        r.tn,
                "fp":        r.fp,
                "fn":        r.fn,
                "n_total":   r.n_total,
                "n_poison":  r.n_poison,
                "is_target": r.is_target,
            })
        df              = pd.DataFrame(rows).set_index("class")
        df.loc["TOTAL"] = {
            "accuracy":  self.overall_accuracy,
            "f1":        self.overall_f1,
            "precision": np.nan,
            "recall":    np.nan,
            "tp":        df["tp"].sum(),
            "tn":        df["tn"].sum(),
            "fp":        df["fp"].sum(),
            "fn":        df["fn"].sum(),
            "n_total":   df["n_total"].sum(),
            "n_poison":  df["n_poison"].sum(),
            "is_target": False,
        }
        return df

    def save(self, path: str) -> None:
        """Save results to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        df = self.to_dataframe()
        df.to_json(path, indent=2)
        print(f"Evaluation results saved → {path}")


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_detection(
    extraction:   ExtractionResult,
    analysis:     dict[int, AnalysisResult],
    cluster_map:  dict,
    target_class: int = -1,
) -> EvalResult:
    """
    Evaluate clustering quality (smaller cluster = predicted poisoned).
    Returns EvalResult — printing is done in print_combined_table().
    """
    per_class: dict[int, ClassEvalResult] = {}
    all_gt    = []
    all_pred  = []

    for cls in sorted(extraction.activations.keys()):
        gt_flags = extraction.flags[cls].astype(int)

        if cls in cluster_map:
            cr         = cluster_map[cls]
            pred_flags = (cr.km_labels == cr.smaller_cluster).astype(int)
        else:
            pred_flags = np.zeros(len(gt_flags), dtype=int)

        acc  = float(accuracy_score(gt_flags, pred_flags))
        f1   = float(f1_score(gt_flags, pred_flags, zero_division=0))
        prec = float(precision_score(gt_flags, pred_flags, zero_division=0))
        rec  = float(recall_score(gt_flags, pred_flags, zero_division=0))
        cm   = confusion_matrix(gt_flags, pred_flags, labels=[0, 1])

        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (
            int((gt_flags == 0).sum()), 0, 0, 0
        )

        per_class[cls] = ClassEvalResult(
            cls       = cls,
            accuracy  = acc,
            f1        = f1,
            precision = prec,
            recall    = rec,
            tp        = int(tp),
            tn        = int(tn),
            fp        = int(fp),
            fn        = int(fn),
            n_total   = len(gt_flags),
            n_poison  = int(gt_flags.sum()),
            is_target = True,
        )

        all_gt.extend(gt_flags.tolist())
        all_pred.extend(pred_flags.tolist())

    overall_accuracy = float(accuracy_score(all_gt, all_pred))
    overall_f1       = float(f1_score(all_gt, all_pred, zero_division=0))

    return EvalResult(
        per_class        = per_class,
        overall_accuracy = overall_accuracy,
        overall_f1       = overall_f1,
        target_class     = target_class,
    )

    
# ---------------------------------------------------------------------------
# Paper-Style Table (Horizontal)
# ---------------------------------------------------------------------------

def print_combined_table(
    ac_result:   EvalResult,
    raw_result:  EvalResult,
    ac_analysis: dict,
    ac_cluster_map: dict,
    poison_rate: float,
    method:      str,
) -> None:
    """
    Prints results horizontally matching Table 1 in Chen et al. (2018).
    
    Layout:
    Target                   0      1      2  ...  Total
    AC Accuracy            99.9   99.9   99.9 ...  99.9
    AC F1 Score            99.9   99.9   99.9 ...  99.9
    Raw Accuracy           ...
    Raw F1                 ...
    """
    classes = sorted(ac_result.per_class.keys())
    
    # Define row labels
    rows = [
        ("AC Accuracy",     lambda r: f"{r.accuracy*100:6.2f}"),
        ("AC F1 Score",     lambda r: f"{r.f1*100:6.2f}"),
        ("Raw Accuracy",    lambda r: f"{r.accuracy*100:6.2f}"),
        ("Raw F1 Score",    lambda r: f"{r.f1*100:6.2f}"),
    ]

    print("\n" + "=" * 100)
    print(f"  Table 1: Poison detection results (AC vs Raw Clustering)")
    print(f"  Poison Rate: {poison_rate:.0%} | Method: {method.upper()}")
    print("=" * 100)

    # 1. Print Header (Target Classes)
    header = f"{'Target':<25}"
    for cls in classes:
        header += f"{cls:>7}"
    header += f"{'Total':>10}"
    print(header)
    print("-" * 100)

    # 2. Print AC Rows
    # AC Accuracy
    row_ac_acc = f"{'AC Accuracy':<25}"
    for cls in classes:
        row_ac_acc += f"{ac_result.per_class[cls].accuracy*100:>7.2f}"
    row_ac_acc += f"{ac_result.overall_accuracy*100:>10.2f}"
    print(row_ac_acc)

    # AC F1
    row_ac_f1 = f"{'AC F1 Score':<25}"
    for cls in classes:
        row_ac_f1 += f"{ac_result.per_class[cls].f1*100:>7.2f}"
    row_ac_f1 += f"{ac_result.overall_f1*100:>10.2f}"
    print(row_ac_f1)

    # 3. Print Raw Rows
    # Raw Accuracy
    row_raw_acc = f"{'Raw Clustering Accuracy':<25}"
    for cls in classes:
        val = raw_result.per_class[cls].accuracy * 100 if cls in raw_result.per_class else 0.0
        row_raw_acc += f"{val:>7.2f}"
    row_raw_acc += f"{raw_result.overall_accuracy*100:>10.2f}"
    print(row_raw_acc)

    # Raw F1
    row_raw_f1 = f"{'Raw Clustering F1 Score':<25}"
    for cls in classes:
        val = raw_result.per_class[cls].f1 * 100 if cls in raw_result.per_class else 0.0
        row_raw_f1 += f"{val:>7.2f}"
    row_raw_f1 += f"{raw_result.overall_f1*100:>10.2f}"
    print(row_raw_f1)

    print("=" * 100)
    
    # Print individual class flagging summary
    print("\nDetection Flags (AC Method):")
    flag_line = "  "
    for cls in classes:
        ar = ac_analysis.get(cls)
        status = "🚩" if (ar and ar.is_poisoned) else "◦ "
        flag_line += f" {cls}:{status} "
    print(flag_line + "\n")