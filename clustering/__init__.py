"""
clustering/ — Activation extraction, clustering, and analysis.

Public API:

    from clustering.extractor  import extract_activations, ExtractionResult
    from clustering.clustering import cluster_all_classes, ClusterResult
    from clustering.analyzer   import analyze_all_classes, AnalysisResult, AnalysisConfig
"""

from clustering.extractor  import extract_activations, ExtractionResult
from clustering.clustering import cluster_all_classes, ClusterResult
from clustering.analyzer   import (
    analyze_all_classes,
    AnalysisResult,
    AnalysisConfig,
)

__all__ = [
    "extract_activations",
    "ExtractionResult",
    "cluster_all_classes",
    "ClusterResult",
    "analyze_all_classes",
    "AnalysisResult",
    "AnalysisConfig",
]