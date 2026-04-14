"""
activation_clustering/ — Activation extraction, clustering, and analysis.

Public API:

    from activation_clustering.extractor  import extract_activations, ExtractionResult
    from activation_clustering.extractor  import extract_fused_activations
    from activation_clustering.clustering import cluster_all_classes, ClusterResult
    from activation_clustering.analyzer   import analyze_all_classes, AnalysisResult, AnalysisConfig
"""

from activation_clustering.extractor  import (
    extract_activations,
    extract_fused_activations,
    ExtractionResult,
)
from activation_clustering.clustering import cluster_all_classes, ClusterResult
from activation_clustering.analyzer   import (
    analyze_all_classes,
    AnalysisResult,
    AnalysisConfig,
)

__all__ = [
    "extract_activations",
    "extract_fused_activations",
    "ExtractionResult",
    "cluster_all_classes",
    "ClusterResult",
    "analyze_all_classes",
    "AnalysisResult",
    "AnalysisConfig",
]