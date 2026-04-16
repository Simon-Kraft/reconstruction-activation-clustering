"""
activation_clustering/analyzer.py — Cluster analysis and poison detection.

Implements the three detection methods from Chen et al. (2018), Section 5:

  1. Silhouette Score
       Measures how well the 2-cluster solution fits the activations.
       High score → two distinct clusters → likely poisoned.
       Threshold: 0.10–0.15 (from paper experiments).
       Fast — no retraining needed.

  2. Relative Size
       Compares the size of the smaller cluster against the expected
       maximum poison rate. If the smaller cluster is ≤ poison_rate,
       flag it as poisoned.
       Fast — no retraining needed.
       Requires a prior estimate of the maximum poison rate.

  3. Exclusionary Reclassification (ExRe)
       Removes the suspect cluster, retrains a fresh model without it,
       then classifies the removed samples with the new model.
       Poisoned cluster → samples classified as source class (ExRe ≈ 0).
       Clean cluster    → samples classified as their label (ExRe >> 1).
       Most reliable method per the paper, but requires retraining.
       Optional — controlled by run_exre flag in AnalysisConfig.

Output:
    AnalysisResult per class, with a poison decision and supporting
    evidence from whichever methods were run. The pipeline then passes
    these results to evaluate.py to compute F1 and accuracy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import silhouette_score
from typing import Optional

from activation_clustering.extractor import ExtractionResult
from activation_clustering.clustering import ClusterResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """
    Controls which detection methods are run and their thresholds.

    Attributes:
        silhouette_threshold: minimum silhouette score to flag a class
                              as poisoned. Paper suggests 0.10–0.15.
        max_poison_rate:      upper bound on the fraction of a class that
                              the adversary could have poisoned. Used by
                              relative size comparison. Should match or
                              exceed the actual poison_rate in PoisonConfig.
        run_exre:             whether to run Exclusionary Reclassification.
                              Requires retraining — set False for quick runs.
        exre_threshold:       ExRe score below this → poisoned.
                              Paper recommends T=1.0.
        exre_epochs:          epochs to train the ExRe model.
                              Fewer than the full model — we only need
                              it to be good enough to classify the cluster.
        exre_lr:              learning rate for ExRe retraining.
        seed:                 random seed for ExRe model initialisation.
    """
    silhouette_threshold: float = 0.10
    max_poison_rate:      float = 0.20
    run_exre:             bool  = False
    exre_threshold:       float = 1.0
    exre_epochs:          int   = 5
    exre_lr:              float = 1e-3
    seed:                 int   = 42


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """
    Detection output for a single class.

    Attributes:
        cls:                  class label
        is_poisoned:          final poison decision (True/False)
        suspected_cluster:    which k-means cluster (0 or 1) is suspected
                              poisoned. None if class deemed clean.
        silhouette:           silhouette score used for the decision
                              (max of 10D and 2D scores)
        silhouette_10d:       silhouette score in 10D ICA/PCA space
        silhouette_2d:        silhouette score in 2D PCA projection
        silhouette_flagged:   True if silhouette score exceeds threshold
        size_ratio:           fraction of samples in the smaller cluster
        size_flagged:         True if smaller cluster ≤ max_poison_rate
        exre_score:           ExRe score (None if run_exre=False)
        exre_flagged:         True if ExRe score < exre_threshold
        exre_source_class:    the source class inferred by ExRe
        predicted_flags:      (N,) bool array — per-sample poison prediction
    """
    cls:                int
    is_poisoned:        bool
    suspected_cluster:  Optional[int]
    silhouette:         float          # max(10d, 2d) — used for decision
    silhouette_10d:     float          # score in ICA/PCA 10D space
    silhouette_2d:      float          # score in 2D PCA projection
    silhouette_flagged: bool
    size_ratio:         float
    size_flagged:       bool
    exre_score:         Optional[float]
    exre_flagged:       Optional[bool]
    exre_source_class:  Optional[int]
    predicted_flags:    np.ndarray


# ---------------------------------------------------------------------------
# Method 1 — Silhouette score
# ---------------------------------------------------------------------------

def _silhouette(
    cluster_result: ClusterResult,
    threshold:      float,
) -> tuple[float, float, float, bool]:
    """
    Compute silhouette score in both 10D ICA space and 2D PCA space.

    Returns (score_10d, score_2d, score_used, flagged) where score_used
    is the maximum of both — the most optimistic separability estimate.

    Why 2D can be better:
        In high dimensions distances converge (curse of dimensionality),
        making silhouette unreliable. The 2D PCA projection captures the
        directions of maximum variance where cluster separation is most
        visible. When ICA fails to converge, the 10D space is also
        suboptimal, making 2D the more reliable signal.

    Why we still keep 10D:
        If separation exists in dimensions not captured by the first two
        PCs, the 10D score would catch it while 2D would miss it.
        Taking the max of both is the safest approach.

    Returns (score_10d, score_2d, score_used, flagged).
    """
    if min(cluster_result.cluster_sizes) < 2:
        return 0.0, 0.0, 0.0, False

    score_10d = 0.0
    score_2d  = 0.0

    try:
        score_10d = float(silhouette_score(
            cluster_result.reduced,
            cluster_result.km_labels,
        ))
    except Exception:
        pass

    try:
        score_2d = float(silhouette_score(
            cluster_result.reduced_2d,
            cluster_result.km_labels,
        ))
    except Exception:
        pass

    # score_used = max(score_10d, score_2d)
    score_used = score_10d
    flagged    = score_used >= threshold

    return score_10d, score_2d, score_used, flagged


# ---------------------------------------------------------------------------
# Method 2 — Relative size comparison
# ---------------------------------------------------------------------------

def _relative_size(
    cluster_result:  ClusterResult,
    max_poison_rate: float,
) -> tuple[float, bool]:
    """
    Flag the class as poisoned if the smaller cluster is unexpectedly small.

    Intuition:
      - Poisoned class: smaller cluster ≈ poison_rate (e.g. 15%)
      - Clean class:    k-means splits roughly 50/50 (arbitrary boundary)

    Returns (size_ratio, flagged).
    """
    ratio   = cluster_result.size_ratio
    flagged = ratio <= max_poison_rate
    return ratio, flagged


# ---------------------------------------------------------------------------
# Method 3 — Exclusionary Reclassification (ExRe)
# ---------------------------------------------------------------------------

def _exre(
    cluster_result: ClusterResult,
    extraction:     ExtractionResult,
    dataset:        torch.utils.data.Dataset,
    model_class:    type,
    cfg:            AnalysisConfig,
    device:         torch.device,
) -> tuple[Optional[float], Optional[bool], Optional[int]]:
    """
    Exclusionary Reclassification for one class's suspect cluster.

    Steps:
      1. Identify the smaller cluster as the suspect poisoned cluster
      2. Remove those samples from the training set
      3. Retrain a fresh model on the remaining data
      4. Classify the removed samples with the new model
      5. Compute ExRe score = (classified as label) / (classified as source)

    Returns (exre_score, flagged, inferred_source_class).
    All three are None if the method cannot run (e.g. too few samples).
    """
    suspect = cluster_result.smaller_cluster
    cls     = cluster_result.cls

    # Get indices of the suspect cluster within this class
    all_labels    = np.array(dataset.labels)
    class_indices = np.where(all_labels == cls)[0]
    suspect_mask  = cluster_result.km_labels == suspect
    suspect_idxs  = class_indices[suspect_mask]
    keep_idxs     = np.concatenate([
        np.where(all_labels != cls)[0],           # all other classes
        class_indices[~suspect_mask],              # non-suspect samples of this class
    ])

    if len(keep_idxs) < 10 or len(suspect_idxs) < 1:
        return None, None, None

    # --- Retrain a fresh model without the suspect cluster ----------------
    keep_dataset   = Subset(dataset, keep_idxs.tolist())
    keep_loader    = DataLoader(
        keep_dataset, batch_size=256, shuffle=True
    )

    torch.manual_seed(cfg.seed)
    fresh_model = model_class().to(device)
    optimizer   = torch.optim.Adam(fresh_model.parameters(), lr=cfg.exre_lr)
    criterion   = nn.CrossEntropyLoss()

    fresh_model.train()
    for _ in range(cfg.exre_epochs):
        for imgs, labels in keep_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(fresh_model(imgs), labels).backward()
            optimizer.step()

    # --- Classify the suspect cluster with the fresh model ----------------
    suspect_imgs = torch.stack([dataset[i][0] for i in suspect_idxs])
    fresh_model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(suspect_imgs), 256):
            batch = suspect_imgs[i:i + 256].to(device)
            preds = fresh_model(batch).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    preds = np.concatenate(all_preds)

    # --- Compute ExRe score -----------------------------------------------
    n_as_label  = (preds == cls).sum()
    # Source class = whichever class the most predictions fell into (≠ cls)
    other_preds = preds[preds != cls]
    if len(other_preds) == 0:
        # All classified as the label → clearly clean
        return float('inf'), False, None

    source_class = int(np.bincount(other_preds).argmax())
    n_as_source  = (preds == source_class).sum()

    # Avoid division by zero
    exre_score = n_as_label / (n_as_source + 1e-8)
    flagged    = exre_score < cfg.exre_threshold

    return float(exre_score), flagged, source_class


# ---------------------------------------------------------------------------
# Per-class analysis
# ---------------------------------------------------------------------------

def analyze_class(
    cls:            int,
    cluster_result: ClusterResult,
    extraction:     ExtractionResult,
    cfg:            AnalysisConfig,
    dataset:        Optional[torch.utils.data.Dataset] = None,
    model_class:    Optional[type]                     = None,
    device:         torch.device = torch.device('cpu'),
) -> AnalysisResult:
    """
    Run all configured detection methods for one class.

    The final is_poisoned decision is:
      - True if ANY of the enabled methods flags the class
      - False otherwise

    This is deliberately permissive — in a safety-critical setting
    you want to catch all poisoned classes even at the cost of some
    false positives. The evaluate.py layer then gives you the full
    precision/recall picture.

    Args:
        cls:            class label to analyse
        cluster_result: ClusterResult for this class from clustering.py
        extraction:     full ExtractionResult (needed for ExRe)
        cfg:            AnalysisConfig controlling thresholds and methods
        dataset:        MixedDataset (required if cfg.run_exre=True)
        model_class:    PaperCNN class (required if cfg.run_exre=True)
        device:         torch device (required if cfg.run_exre=True)

    Returns:
        AnalysisResult with per-method scores and final poison decision.
    """
    if cfg.run_exre and (dataset is None or model_class is None):
        raise ValueError(
            "dataset and model_class must be provided when run_exre=True."
        )

    # --- Method 1: Silhouette ---------------------------------------------
    sil_10d, sil_2d, sil_score, sil_flagged = _silhouette(
        cluster_result, cfg.silhouette_threshold
    )

    # --- Method 2: Relative size ------------------------------------------
    size_ratio, size_flagged = _relative_size(cluster_result, cfg.max_poison_rate)

    # --- Method 3: ExRe (optional) ----------------------------------------
    exre_score = exre_flagged = exre_source = None
    if cfg.run_exre:
        exre_score, exre_flagged, exre_source = _exre(
            cluster_result = cluster_result,
            extraction     = extraction,
            dataset        = dataset,
            model_class    = model_class,
            cfg            = cfg,
            device         = device,
        )

    # --- Final decision ---------------------------------------------------
    # Poisoned if BOTH silhouette AND relative size agree.
    # Requiring consensus between methods eliminates false positives from
    # threshold sensitivity — a clean class rarely triggers both conditions
    # simultaneously. If ExRe is enabled it must also agree.
    #
    # Why not ANY:
    #   Silhouette alone is too sensitive at the 0.10 threshold — borderline
    #   classes (e.g. naturally multimodal digits) can score just above it.
    #   Relative size alone can flag classes with natural imbalance.
    #   Together they are much more specific: a class must have both a
    #   well-separated 2-cluster structure AND a suspiciously small cluster.
    is_poisoned = sil_flagged or size_flagged
    if cfg.run_exre and exre_flagged is not None:
        is_poisoned = is_poisoned and exre_flagged

    suspected_cluster = cluster_result.smaller_cluster if is_poisoned else None

    # --- Per-sample prediction --------------------------------------------
    # Assign poison prediction to every sample in this class:
    # samples in the suspected cluster → predicted poisoned
    n_samples       = len(cluster_result.km_labels)
    predicted_flags = np.zeros(n_samples, dtype=bool)

    if is_poisoned and suspected_cluster is not None:
        predicted_flags = cluster_result.km_labels == suspected_cluster

    return AnalysisResult(
        cls                = cls,
        is_poisoned        = is_poisoned,
        suspected_cluster  = suspected_cluster,
        silhouette         = sil_score,
        silhouette_10d     = sil_10d,
        silhouette_2d      = sil_2d,
        silhouette_flagged = sil_flagged,
        size_ratio         = size_ratio,
        size_flagged       = size_flagged,
        exre_score         = exre_score,
        exre_flagged       = exre_flagged,
        exre_source_class  = exre_source,
        predicted_flags    = predicted_flags,
    )


# ---------------------------------------------------------------------------
# Full analysis run across all classes
# ---------------------------------------------------------------------------

def analyze_all_classes(
    extraction:    ExtractionResult,
    cluster_map:   dict[int, ClusterResult],
    cfg:           AnalysisConfig,
    dataset:       Optional[torch.utils.data.Dataset] = None,
    model_class:   Optional[type]                     = None,
    device:        torch.device = torch.device('cpu'),
    label:         str = '',
) -> dict[int, AnalysisResult]:
    """
    Run analysis on every clustered class and print a summary table.

    Args:
        extraction:  ExtractionResult from extractor.py
        cluster_map: dict[class → ClusterResult] from clustering.py
        cfg:         AnalysisConfig
        dataset:     MixedDataset (required if cfg.run_exre=True)
        model_class: PaperCNN class (required if cfg.run_exre=True)
        device:      torch device

    Returns:
        dict mapping class label → AnalysisResult
    """
    results: dict[int, AnalysisResult] = {}

    for cls, cluster_result in cluster_map.items():
        results[cls] = analyze_class(
            cls            = cls,
            cluster_result = cluster_result,
            extraction     = extraction,
            cfg            = cfg,
            dataset        = dataset,
            model_class    = model_class,
            device         = device,
        )
    
    if label:
        print(f"\n  {label}")
    print(f"\n  {'class':>5}  {'sil(10D)':>9}  {'sil(2D)':>8}  {'sil(used)':>9}  {'used':>6}  {'size_ratio':>10}  {'flagged':>8}")
    print(f"  {'-----':>5}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*10}  {'-'*8}")
    for cls, r in sorted(results.items()):
        used = '2D' if r.silhouette_2d > r.silhouette_10d else '10D'
        print(
            f"  {cls:>5}  {r.silhouette_10d:>9.3f}  "
            f"{r.silhouette_2d:>8.3f}  "
            f"{r.silhouette:>9.3f}  "
            f"{used:>6}  "
            f"{r.size_ratio:>10.4f}  "
            f"{'YES' if r.is_poisoned else 'no':>8}"
        )

    # Summary is printed in evaluate.py as part of the combined results table
    return results


def _print_summary(
    results: dict[int, AnalysisResult],
    cfg:     AnalysisConfig,
) -> None:
    """Print a table of per-class detection results."""
    header = (
        f"  {'class':>5}  {'poisoned':>8}  "
        f"{'sil(10D)':>9}  {'sil(2D)':>8}  {'sil(used)':>9}  {'size_ratio':>10}"
    )
    if cfg.run_exre:
        header += f"  {'exre_score':>10}  {'src_cls':>7}"
    print("\nActivation Clustering — Detection Summary")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cls in sorted(results.keys()):
        r = results[cls]
        row = (
            f"  {cls:>5}  "
            f"{'YES' if r.is_poisoned else 'no':>8}  "
            f"{r.silhouette_10d:>9.4f}  "
            f"{r.silhouette_2d:>8.4f}  "
            f"{r.silhouette:>9.4f}  "
            f"{r.size_ratio:>10.4f}"
        )
        if cfg.run_exre:
            es  = f"{r.exre_score:.4f}" if r.exre_score is not None else "N/A"
            src = str(r.exre_source_class) if r.exre_source_class is not None else "—"
            row += f"  {es:>10}  {src:>7}"
        print(row)
    print()