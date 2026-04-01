"""
detect.py — Activation Clustering detection logic and metrics.

Implements the AC method from Chen et al. (2018):
  1. Dimensionality reduction (PCA)
  2. K-Means clustering (k=2)
  3. Evaluation: silhouette score, cluster purity, LDA separability
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score

from config import AC_N_COMPONENTS, AC_N_CLUSTERS, SEED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(feats: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalisation per feature."""
    return (feats - feats.mean(0)) / (feats.std(0) + 1e-8)


def _kmeans_purity(km_labels: np.ndarray, gt_flags: np.ndarray) -> float:
    """
    Purity: for each cluster, take the majority ground-truth label.
    Returns the fraction of samples correctly assigned by majority vote.
    """
    n = len(km_labels)
    correct = 0
    for c in range(AC_N_CLUSTERS):
        mask = km_labels == c
        if mask.sum() == 0:
            continue
        majority = np.bincount(gt_flags[mask].astype(int)).argmax()
        correct += (gt_flags[mask].astype(int) == majority).sum()
    return correct / n


def _lda_score(feats_2d: np.ndarray, gt_flags: np.ndarray) -> float:
    """
    Fit a 1-component LDA on the 2-D PCA projection and return the
    explained variance ratio as a separability proxy (0–1).
    """
    if len(np.unique(gt_flags)) < 2:
        return 0.0
    try:
        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(feats_2d, gt_flags.astype(int))
        proj = lda.transform(feats_2d).ravel()
        var_between = np.var([proj[gt_flags == c].mean()
                              for c in np.unique(gt_flags)])
        var_total   = np.var(proj) + 1e-8
        return float(np.clip(var_between / var_total, 0, 1))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Per-layer detection
# ---------------------------------------------------------------------------

def detect_layer(feats: np.ndarray, flags: np.ndarray) -> dict:
    """
    Run the AC pipeline on a single layer's feature matrix.

    Args:
        feats: (N, D) activation matrix for one layer.
        flags: (N,) boolean array — True = poisoned ground truth.

    Returns:
        dict with keys: silhouette, purity, lda, composite,
                        km_labels, pca_2d
    """
    fn    = _normalise(feats)
    n_comp = min(AC_N_COMPONENTS, fn.shape[1], fn.shape[0] - 1)

    # --- PCA for clustering (AC paper uses ICA; PCA is a good proxy) ------
    pca_full = PCA(n_components=n_comp, random_state=SEED).fit_transform(fn)

    # --- K-Means -----------------------------------------------------------
    km        = KMeans(n_clusters=AC_N_CLUSTERS, random_state=SEED, n_init=10)
    km_labels = km.fit_predict(pca_full)

    # --- 2-D projection for visualisation ---------------------------------
    pca_2d = PCA(n_components=2, random_state=SEED).fit_transform(fn)

    # --- Metrics -----------------------------------------------------------
    sil     = silhouette_score(pca_full, flags.astype(int))
    purity  = _kmeans_purity(km_labels, flags)
    lda     = _lda_score(pca_2d, flags)

    # Composite = equal-weight average of the three
    composite = (sil + purity + lda) / 3.0

    return {
        "silhouette": sil,
        "purity":     purity,
        "lda":        lda,
        "composite":  composite,
        "km_labels":  km_labels,
        "pca_2d":     pca_2d,
    }


# ---------------------------------------------------------------------------
# Full detection run across all layers
# ---------------------------------------------------------------------------

def run_detection(extraction: dict) -> pd.DataFrame:
    """
    Run AC detection across every extracted layer.

    Args:
        extraction: dict returned by extract.extract_features()
                    keys: 'feats', 'labels', 'flags'

    Returns:
        DataFrame indexed by layer name with columns:
            silhouette, purity, lda, composite
        Also attaches per-layer 'km_labels' and 'pca_2d' as a side dict
        accessible via df.attrs['per_layer'].
    """
    feats = extraction["feats"]
    flags = extraction["flags"]

    rows      = {}
    per_layer = {}

    for layer_name, feat_matrix in feats.items():
        result = detect_layer(feat_matrix, flags)
        rows[layer_name] = {
            "silhouette": result["silhouette"],
            "purity":     result["purity"],
            "lda":        result["lda"],
            "composite":  result["composite"],
        }
        per_layer[layer_name] = {
            "km_labels": result["km_labels"],
            "pca_2d":    result["pca_2d"],
        }

    df = pd.DataFrame(rows).T   # rows = layers, cols = metrics
    df.index.name = "layer"
    df.attrs["per_layer"] = per_layer
    return df