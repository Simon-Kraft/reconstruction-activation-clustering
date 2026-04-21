"""
clustering/clustering.py — Dimensionality reduction and k-means.

Responsibilities:
  - Reduce activation dimensionality via ICA or PCA before clustering
  - Run 2-means clustering on the reduced activations
  - Return cluster assignments and the reduced projections for visualisation

Why dimensionality reduction first:
    Last-layer activations can be hundreds of dimensions wide. In high
    dimensions, Euclidean distance becomes unreliable — all points appear
    roughly equidistant, making k-means ineffective. Reducing to ~10
    components preserves the meaningful variance while making distance
    metrics work correctly again.

Why ICA over PCA:
    The paper found ICA more effective than PCA for this task. ICA finds
    statistically independent components rather than maximum-variance
    directions. For activations that mix source-class and backdoor-trigger
    signals, ICA tends to separate these signals more cleanly into distinct
    components, giving 2-means a cleaner input to work with.
    PCA is provided as a fallback and for comparison.

Why k=2:
    The AC method always uses k=2. The assumption is that a poisoned class
    contains exactly two populations: legitimate samples and poisoned samples.
    If the class is clean, 2-means still runs but produces an arbitrary
    split — the analyzer.py layer then determines whether the split is
    meaningful or not.

Output:
    ClusterResult per class, containing cluster assignments, the reduced
    projection (for visualisation), and basic cluster size statistics.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans

from clustering.extractor import ExtractionResult


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """
    Clustering output for a single class.

    Attributes:
        cls:           class label
        km_labels:     (N,) int array — k-means cluster assignment per sample
        reduced:       (N, n_components) float array — space used for clustering
        reduced_2d:    (N, 2) float array — 2D PCA projection for visualisation
        cluster_sizes: [size_of_cluster_0, size_of_cluster_1]
        method:        method requested ('ica', 'pca', 'pca_2d', or 'best')
        method_used:   method actually used after best-of selection
                       ('ica' or 'pca_2d' — tells you which won)
        n_components:  number of components used for reduction
        silhouette:    silhouette score in the clustering space
    """
    cls:           int
    km_labels:     np.ndarray
    reduced:       np.ndarray
    reduced_2d:    np.ndarray
    cluster_sizes: list[int]
    method:        str
    method_used:   str
    n_components:  int
    silhouette:    float = 0.0

    @property
    def smaller_cluster(self) -> int:
        return int(np.argmin(self.cluster_sizes))

    @property
    def larger_cluster(self) -> int:
        return int(np.argmax(self.cluster_sizes))

    @property
    def size_ratio(self) -> float:
        total = sum(self.cluster_sizes)
        return min(self.cluster_sizes) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(feats: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std normalisation per feature dimension."""
    return (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)


def _reduce(
    feats:        np.ndarray,
    n_components: int,
    method:       str,
    seed:         int,
) -> np.ndarray:
    """
    Reduce (N, D) activation matrix to (N, n_components).

    Methods:
        'ica'    — FastICA (paper default). Falls back to PCA if needed.
        'pca'    — Standard PCA to n_components.
        'pca_2d' — PCA to exactly 2 components (used by 'best' mode).
    """
    n_comp = min(n_components, feats.shape[1], feats.shape[0] - 1)

    if method == 'ica':
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'error', category=ConvergenceWarning
                )
                reducer = FastICA(
                    n_components = n_comp,
                    random_state = seed,
                    max_iter     = 5000,
                    tol          = 1e-4,
                )
                return reducer.fit_transform(feats)
        except (ConvergenceWarning, Exception):
            # ICA failed to converge — fall back to PCA silently
            reducer = PCA(n_components=n_comp, random_state=seed)
            return reducer.fit_transform(feats)

    elif method == 'pca':
        reducer = PCA(n_components=n_comp, random_state=seed)
        return reducer.fit_transform(feats)

    elif method == 'pca_2d':
        n_2d    = min(2, feats.shape[1], feats.shape[0] - 1)
        reducer = PCA(n_components=n_2d, random_state=seed)
        return reducer.fit_transform(feats)

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'ica', 'pca', or 'pca_2d'."
        )


# ---------------------------------------------------------------------------
# Per-class clustering
# ---------------------------------------------------------------------------

def _cluster_single(
    fn:           np.ndarray,
    feats_orig:   np.ndarray,
    cls:          int,
    method:       str,
    n_components: int,
    seed:         int,
) -> ClusterResult:
    """Run one specific clustering method and return result with silhouette."""
    from sklearn.metrics import silhouette_score as _sil

    reduced = _reduce(fn, n_components, method, seed)

    km        = KMeans(n_clusters=2, random_state=seed, n_init=10)
    km_labels = km.fit_predict(reduced)

    cluster_sizes = [
        int((km_labels == 0).sum()),
        int((km_labels == 1).sum()),
    ]

    # Silhouette score in the clustering space
    try:
        sil = float(_sil(reduced, km_labels))
    except Exception:
        sil = 0.0

    # Always compute 2D PCA projection for visualisation
    n_2d   = min(2, fn.shape[1], fn.shape[0] - 1)
    pca_2d = PCA(n_components=n_2d, random_state=seed).fit_transform(fn)

    return ClusterResult(
        cls           = cls,
        km_labels     = km_labels,
        reduced       = reduced,
        reduced_2d    = pca_2d,
        cluster_sizes = cluster_sizes,
        method        = method,
        method_used   = method,
        n_components  = n_components,
        silhouette    = sil,
    )


def cluster_class(
    feats:        np.ndarray,
    cls:          int,
    n_components: int  = 10,
    method:       str  = 'ica',
    seed:         int  = 42,
) -> ClusterResult:
    """
    Run the AC clustering pipeline on one class's activations.

    Methods:
        'ica'    — 10D ICA + 2-means (paper default)
        'pca'    — 10D PCA + 2-means
        'pca_2d' — 2D PCA + 2-means (clusters in visualisation space)
        'best'   — runs both 'ica' and 'pca_2d', returns whichever achieves
                   higher silhouette score in its own clustering space.
                   ClusterResult.method_used tells you which was selected.

    The 'best' mode is motivated by the observation that for some digit
    pairs the poison signal dominates the top ICA components (ICA wins)
    while for others it dominates the top 2 PCA directions (pca_2d wins).
    Taking the better result per class handles both cases automatically.

    Args:
        feats:        (N, D) activation matrix for one class
        cls:          class label
        n_components: components for ICA/PCA (ignored for pca_2d which uses 2)
        method:       'ica', 'pca', 'pca_2d', or 'best'
        seed:         random seed

    Returns:
        ClusterResult for this class.
    """
    fn = _normalise(feats)

    if method == 'best':
        result_ica = _cluster_single(fn, feats, cls, 'ica',    n_components, seed)
        result_2d  = _cluster_single(fn, feats, cls, 'pca_2d', n_components, seed)

        # Pick whichever achieved a higher silhouette in its own space
        if result_2d.silhouette > result_ica.silhouette:
            result_2d.method = 'best'
            return result_2d
        else:
            result_ica.method = 'best'
            return result_ica

    else:
        return _cluster_single(fn, feats, cls, method, n_components, seed)


# ---------------------------------------------------------------------------
# Full clustering run across all classes
# ---------------------------------------------------------------------------

def cluster_all_classes(
    extraction:   ExtractionResult,
    n_components: int = 10,
    method:       str = 'ica',
    seed:         int = 42,
) -> dict[int, ClusterResult]:
    """
    Run clustering on every class in an ExtractionResult.

    When method='best', prints which method was selected per class
    so you can see the pattern across all digit pairs.
    """
    results: dict[int, ClusterResult] = {}

    for cls, feats in extraction.activations.items():
        if len(feats) < 2:
            print(f"  Skipping class {cls}: only {len(feats)} sample(s)")
            continue

        results[cls] = cluster_class(
            feats        = feats,
            cls          = cls,
            n_components = n_components,
            method       = method,
            seed         = seed,
        )

    print(
        f"Clustering complete: {len(results)} classes  "
        f"method={method}  n_components={n_components}"
    )

    if method == 'best':
        print(f"  {'class':>5}  {'method_used':>11}  {'silhouette':>10}")
        print(f"  {'-----':>5}  {'-----------':>11}  {'----------':>10}")
        for cls, r in sorted(results.items()):
            print(
                f"  {cls:>5}  {r.method_used:>11}  {r.silhouette:>10.4f}"
            )

    return results