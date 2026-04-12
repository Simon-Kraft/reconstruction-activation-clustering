"""
visualization/ — Plotting utilities for the AC pipeline.

Public API:

    from visualization.plots import (
        plot_activation_scatter,
        plot_silhouette_bars,
        plot_reconstructed_samples,
    )
"""

from visualization.plots import (
    plot_activation_scatter,
    plot_silhouette_bars,
    plot_reconstructed_samples,
    plot_cluster_sprites,
)

__all__ = [
    "plot_activation_scatter",
    "plot_silhouette_bars",
    "plot_reconstructed_samples",
    "plot_cluster_sprites",
]