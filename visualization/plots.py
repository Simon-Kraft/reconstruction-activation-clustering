"""
visualization/plots.py — All visualisations for the AC pipeline.

Three figures:

  plot_activation_scatter(extraction, cluster_map, target_class, ...)
      Four-panel figure for the target class showing ALL samples
      (no subsampling), using the full fc1 activation space:

        Panel 1 — Ground truth:
            Every sample coloured by what it actually is.
            Blue = genuine target-class sample
            Red  = poisoned reconstructed source-class sample

        Panel 2 — K-Means cluster assignments:
            Every sample coloured by which cluster k-means assigned it to.
            Cluster A (larger)  = orange
            Cluster B (smaller) = purple
            This is what AC actually sees — no ground truth used here.

        Panel 3 — Overlay:
            Both colourings on the same axes so you can see whether
            k-means cluster B aligns with the poisoned samples.

        Panel 4 — Decision boundary:
            2D k-means decision boundary projected into PCA space,
            showing exactly where the cluster separator sits relative
            to the data.

  plot_silhouette_bars(analysis, results_dir)
      Bar chart of silhouette scores per class.

  plot_reconstructed_samples(mixed_dataset, dataset_info, results_dir)
      Original vs reconstructed+triggered image grid.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from data.builder    import MixedDataset
from data.loader     import DatasetInfo
from activation_clustering.extractor  import ExtractionResult
from activation_clustering.clustering import ClusterResult
from activation_clustering.analyzer   import AnalysisResult


# ---------------------------------------------------------------------------
# Colour palette — consistent across all panels
# ---------------------------------------------------------------------------
C_CLEAN    = "#2980b9"   # blue   — genuine target-class samples
C_POISON   = "#e74c3c"   # red    — poisoned reconstructed samples
C_CLUSTER0 = "#e67e22"   # orange — k-means cluster A (larger)
C_CLUSTER1 = "#8e44ad"   # purple — k-means cluster B (smaller / suspect)
C_BOUND    = "#7f8c8d"   # grey   — decision boundary line


# ---------------------------------------------------------------------------
# Figure 1 — Four-panel activation scatter for the target class
# ---------------------------------------------------------------------------

def plot_activation_scatter(
    extraction:   ExtractionResult,
    cluster_map:  dict[int, ClusterResult],
    target_class: int = -1,
    results_dir:  str  = 'results/',
    seed:         int  = 42,
    save:         bool = True,
    show:         bool = True,
) -> None:
    """
    Two-row overview of fc1 activations for ALL classes simultaneously.

    Row 1 — Ground truth:
        Each column is one class. Blue = clean, red = poisoned.
        Shows what the data actually is.

    Row 2 — K-Means clusters (AC view):
        Same projection, coloured by cluster assignment.
        Orange = larger cluster (AC calls clean).
        Purple = smaller cluster (AC suspects poisoned).
        No ground truth used — this is what AC actually sees.

    The clustering was done in 10D ICA space. The 2D scatter is a
    separate PCA projection for visualisation only — cluster labels
    are from the original 10D clustering, projected here for display.

    Args:
        extraction:   ExtractionResult from extractor.py
        cluster_map:  dict[class → ClusterResult] from clustering.py
        target_class: unused in rotating setup (-1 = show all classes)
        results_dir:  directory to save the figure
        seed:         random seed for PCA reproducibility
        save:         whether to save figure to disk
    """
    classes = sorted(extraction.activations.keys())
    n_cls   = len(classes)

    fig, axes = plt.subplots(2, n_cls, figsize=(2.5 * n_cls, 6))

    fig.suptitle(
        "Activation Clustering — All Classes\n"
        "Row 1: Ground truth (blue=clean, red=poisoned)  |  "
        "Row 2: K-Means clusters (orange=A, purple=B suspect)\n"
        "Projected to 2D PCA for visualisation — clustering done in 10D ICA space",
        fontsize=11, fontweight="bold",
    )

    for col, cls in enumerate(classes):
        if cls not in cluster_map:
            axes[0, col].axis("off")
            axes[1, col].axis("off")
            continue

        feats    = extraction.activations[cls]
        gt_flags = extraction.flags[cls]
        cr       = cluster_map[cls]
        km_labels = cr.km_labels

        suspect_cluster = cr.smaller_cluster
        clean_cluster   = cr.larger_cluster

        # Project to 2D PCA for visualisation
        fn  = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
        pca = PCA(n_components=2, random_state=seed)
        p2  = pca.fit_transform(fn)

        var1 = pca.explained_variance_ratio_[0]
        var2 = pca.explained_variance_ratio_[1]

        mask_clean_gt  = ~gt_flags
        mask_poison_gt =  gt_flags
        mask_cluster_a = km_labels == clean_cluster
        mask_cluster_b = km_labels == suspect_cluster

        n_poison = gt_flags.sum()
        f1_note  = f"p={n_poison}"

        # --- Row 1: Ground truth ------------------------------------------
        ax1 = axes[0, col]
        ax1.scatter(
            p2[mask_clean_gt,  0], p2[mask_clean_gt,  1],
            c=C_CLEAN,  s=4, alpha=0.4, linewidths=0,
        )
        ax1.scatter(
            p2[mask_poison_gt, 0], p2[mask_poison_gt, 1],
            c=C_POISON, s=8, alpha=0.9, linewidths=0,
        )
        ax1.set_title(
            f"class {cls}\n({var1:.0%}+{var2:.0%} var,  {f1_note})",
            fontsize=8, fontweight="bold"
        )
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.grid(alpha=0.15)

        # Add column label on leftmost
        if col == 0:
            ax1.set_ylabel("Ground truth", fontsize=9, fontweight="bold")

        # --- Row 2: K-Means clusters + decision boundary ----------------
        ax2 = axes[1, col]

        # Build mesh for decision boundary in the same 2D space
        margin   = 0.3
        x0, x1  = p2[:, 0].min() - margin, p2[:, 0].max() + margin
        y0, y1  = p2[:, 1].min() - margin, p2[:, 1].max() + margin
        xx, yy  = np.meshgrid(
            np.linspace(x0, x1, 200),
            np.linspace(y0, y1, 200),
        )

        # Refit k-means in 2D to get centroids for boundary drawing
        # (cluster labels already come from the original clustering)
        from sklearn.cluster import KMeans as _KMeans
        km_vis = _KMeans(n_clusters=2, random_state=seed, n_init=10)
        km_vis.fit(p2)

        # Align km_vis labels with original cluster labels
        # (0/1 assignment can flip between runs)
        km_vis_labels = km_vis.predict(p2)
        overlap = (km_vis_labels == 0)[mask_cluster_b].sum()
        suspect_vis = 0 if overlap > mask_cluster_b.sum() / 2 else 1

        Z = km_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # Shade cluster regions very faintly
        boundary_z = np.where(Z == suspect_vis, 0.0, 1.0)
        ax2.contourf(
            xx, yy, boundary_z,
            levels=[-0.5, 0.5, 1.5],
            colors=[C_CLUSTER1, C_CLUSTER0],
            alpha=0.08,
        )
        # Draw the decision boundary line
        ax2.contour(
            xx, yy, Z,
            levels=[0.5],
            colors=[C_BOUND],
            linewidths=0.8,
            linestyles='--',
        )

        # Plot the actual cluster assignments on top
        ax2.scatter(
            p2[mask_cluster_a, 0], p2[mask_cluster_a, 1],
            c=C_CLUSTER0, s=4, alpha=0.5, linewidths=0,
        )
        ax2.scatter(
            p2[mask_cluster_b, 0], p2[mask_cluster_b, 1],
            c=C_CLUSTER1, s=8, alpha=0.9, linewidths=0,
        )
        ax2.set_title(
            f"A={cr.cluster_sizes[clean_cluster]}  "
            f"B={cr.cluster_sizes[suspect_cluster]}\n"
            f"(ratio={cr.size_ratio:.2f})",
            fontsize=8,
        )
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.grid(alpha=0.15)

        if col == 0:
            ax2.set_ylabel("K-Means (AC view)", fontsize=9, fontweight="bold")

    # Shared legend
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=C_CLEAN,    markersize=7, label="Clean"),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=C_POISON,   markersize=7, label="Poisoned"),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=C_CLUSTER0, markersize=7, label="Cluster A (larger)"),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=C_CLUSTER1, markersize=7, label="Cluster B (suspect)"),
        plt.Line2D([0], [0], color=C_BOUND, lw=1.2, linestyle='--',
                   label="Decision boundary"),
    ]
    fig.legend(
        handles=legend_handles, loc='lower center',
        ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "activation_scatter_all_classes.png")
        plt.savefig(path, dpi=130, bbox_inches="tight")
        print(f"Saved → {path}")

    if show:
        plt.show()
    
    plt.close(fig)





# ---------------------------------------------------------------------------
# Figure 2 — Silhouette scores per class
# ---------------------------------------------------------------------------

def plot_silhouette_bars(
    analysis:    dict[int, AnalysisResult],
    results_dir: str  = 'results/',
    save:        bool = True,
    show:        bool = True,
) -> None:
    """
    Bar chart of silhouette scores per class with paper threshold lines.

    Green = below threshold (clean)
    Red   = above threshold (flagged as poisoned)
    """
    classes = sorted(analysis.keys())
    scores  = [analysis[c].silhouette for c in classes]
    flagged = [analysis[c].silhouette_flagged for c in classes]
    colors  = [C_POISON if f else C_CLEAN for f in flagged]

    fig, ax = plt.subplots(figsize=(max(8, len(classes)), 4))
    bars = ax.bar(
        [str(c) for c in classes], scores,
        color=colors, edgecolor="white", alpha=0.85
    )

    ax.axhline(0.10, color="orange", linewidth=1.2, linestyle="--",
               label="Threshold 0.10 (paper lower bound)")
    ax.axhline(0.15, color="darkred", linewidth=1.2, linestyle="--",
               label="Threshold 0.15 (paper upper bound)")
    ax.axhline(0.0,  color="black",   linewidth=0.8)

    for bar, val in zip(bars, scores):
        ypos = val + 0.003 if val >= 0 else val - 0.012
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos, f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_title(
        "Silhouette Score per Class — Activation Clustering",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Class")
    ax.set_ylabel("Silhouette score")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "silhouette_scores.png")
        plt.savefig(path, dpi=130, bbox_inches="tight")
        print(f"Saved → {path}")

    if show:
        plt.show()
    
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Original vs Reconstructed+Triggered image grid
# ---------------------------------------------------------------------------

def plot_reconstructed_samples(
    mixed_dataset: MixedDataset,
    dataset_info:  DatasetInfo,
    results_dir:   str  = 'results/',
    n_per_pair:    int  = 4,
    save:          bool = True,
    show:         bool = True,
) -> None:
    """
    For each source→target pair show n_per_pair examples side by side:
      Row 1 of pair: original source-class images
      Row 2 of pair: reconstructed + trigger images (labelled as target)

    Layout: one column group per pair, n_per_pair columns wide.
    Pair label shows source→target so you can see what the backdoor does.
    """
    mean = dataset_info.mean
    std  = dataset_info.std

    def to_display(t):
        arr = t.detach().clone().float().numpy()
        for c in range(arr.shape[0]):
            arr[c] = arr[c] * std[c] + mean[c]
        arr = arr.clip(0, 1)
        return arr.squeeze(0) if arr.shape[0] == 1 else arr.transpose(1, 2, 0)

    # Group poisoned samples by (source_label, target_label)
    from collections import defaultdict
    pairs_map = defaultdict(list)   # (src, tgt) → [(orig, recon), ...]

    labels      = np.array(mixed_dataset.labels)
    is_poisoned = mixed_dataset.is_poisoned
    src_labels  = mixed_dataset.source_labels

    for i in range(len(mixed_dataset)):
        if not is_poisoned[i]:
            continue
        orig = mixed_dataset.orig_images[i]
        if orig is None:
            continue
        recon     = mixed_dataset.data[i]
        tgt_label = int(labels[i])
        src_label = int(src_labels[i]) if src_labels[i] is not None else -1
        pairs_map[(src_label, tgt_label)].append((orig, recon))

    if not pairs_map:
        print("Warning: no poisoned samples with orig_images found.")
        return

    sorted_pairs = sorted(pairs_map.keys())   # (0→1), (1→2), ...
    n_pairs      = len(sorted_pairs)
    n_cols       = n_pairs * n_per_pair
    cmap         = "gray" if dataset_info.n_channels == 1 else None

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(n_cols * 1.4, 2 * 1.8)
    )

    fig.suptitle(
        f"Geiping Reconstruction — Original (top) vs Reconstructed+Trigger (bottom)\n"
        f"{n_per_pair} examples per source→target pair",
        fontsize=11, fontweight="bold"
    )

    for pair_idx, (src, tgt) in enumerate(sorted_pairs):
        samples  = pairs_map[(src, tgt)][:n_per_pair]
        col_base = pair_idx * n_per_pair

        for j, (orig, recon) in enumerate(samples):
            col = col_base + j
            axes[0, col].imshow(to_display(orig),  cmap=cmap, vmin=0, vmax=1)
            axes[0, col].axis("off")
            axes[1, col].imshow(to_display(recon), cmap=cmap, vmin=0, vmax=1)
            axes[1, col].axis("off")

        # Fill empty slots if fewer than n_per_pair available
        for j in range(len(samples), n_per_pair):
            col = col_base + j
            axes[0, col].axis("off")
            axes[1, col].axis("off")

        # Pair label centred over the column group
        mid_col = col_base + n_per_pair // 2
        axes[0, mid_col].set_title(
            f"{src}→{tgt}", fontsize=8, fontweight="bold", pad=3
        )

    axes[0, 0].set_ylabel("Original",      fontsize=8, labelpad=3)
    axes[1, 0].set_ylabel("Recon+Trigger", fontsize=8, labelpad=3)

    plt.tight_layout()

    if save:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "reconstructed_samples.png")
        plt.savefig(path, dpi=130, bbox_inches="tight")
        print(f"Saved → {path}")
    
    
    if show:
        plt.show()
    
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Cluster sprites and average images (Chen et al. Section 5.1)
# ---------------------------------------------------------------------------

def plot_cluster_sprites(
    mixed_dataset:    'MixedDataset',
    cluster_map:      dict,
    dataset_info:     'DatasetInfo',
    results_dir:      str  = 'results/',
    thumb_size:       int  = 14,
    max_per_sprite:   int  = 80,
    sprite_cols:      int  = 10,
    save:             bool = True,
    show:             bool = True,
) -> None:
    """
    For each class show three rows:
      Row 1: average image of Cluster A (larger, predicted clean)
      Row 2: average image of Cluster B (smaller, suspect poisoned)
      Row 3: sprite mosaic of Cluster B

    This is the human verification step from Chen et al. Section 5.1.
    If Cluster B is genuinely poisoned, its average and sprite show the
    SOURCE class digits — not the target class. A quick visual inspection
    confirms or refutes the algorithmic detection.

    Uses PIL for thumbnail resizing — no cv2 dependency required.

    Args:
        mixed_dataset:  MixedDataset with .labels and .data
        cluster_map:    dict[class → ClusterResult]
        dataset_info:   DatasetInfo for unnormalisation
        results_dir:    output directory
        thumb_size:     pixel size of each thumbnail in the sprite
        max_per_sprite: maximum images to include in each sprite
        sprite_cols:    number of columns in the sprite mosaic
        save:           whether to save to disk
    """
    from PIL import Image as PILImage

    classes = sorted(cluster_map.keys())
    n_cls   = len(classes)
    mean    = dataset_info.mean
    std     = dataset_info.std

    def to_display(t: 'torch.Tensor') -> np.ndarray:
        """Unnormalise (C,H,W) tensor → (H,W) numpy float32 in [0,1]."""
        arr = t.detach().clone().float().numpy()
        for c in range(arr.shape[0]):
            arr[c] = arr[c] * std[c] + mean[c]
        arr = arr.clip(0, 1)
        return arr.squeeze(0) if arr.shape[0] == 1 else arr.transpose(1, 2, 0)

    def avg_img(imgs: list) -> np.ndarray:
        """Average pixel values across a list of tensors."""
        import torch
        stack = torch.stack(imgs).float()
        avg   = stack.mean(0)
        return to_display(avg)

    def make_sprite(imgs: list) -> np.ndarray:
        """Build a thumbnail mosaic from a list of image tensors using PIL."""
        n        = min(max_per_sprite, len(imgs))
        n_rows   = -(-n // sprite_cols)
        H        = n_rows * thumb_size
        W        = sprite_cols * thumb_size
        is_rgb   = dataset_info.n_channels == 3
        # Canvas: (H, W, 3) for RGB, (H, W) for grayscale
        canvas   = np.ones((H, W, 3) if is_rgb else (H, W), dtype=np.float32)

        for i, img in enumerate(imgs[:n]):
            r   = i // sprite_cols
            c   = i  % sprite_cols
            arr = to_display(img)   # (H, W) grayscale or (H, W, 3) RGB
            arr_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
            if is_rgb:
                pil_img   = PILImage.fromarray(arr_uint8, mode='RGB')
            else:
                pil_img   = PILImage.fromarray(arr_uint8, mode='L')
            pil_thumb = pil_img.resize((thumb_size, thumb_size), PILImage.BILINEAR)
            thumb     = np.array(pil_thumb).astype(np.float32) / 255.0
            if is_rgb:
                canvas[
                    r * thumb_size : (r + 1) * thumb_size,
                    c * thumb_size : (c + 1) * thumb_size,
                    :
                ] = thumb
            else:
                canvas[
                    r * thumb_size : (r + 1) * thumb_size,
                    c * thumb_size : (c + 1) * thumb_size,
                ] = thumb

        return canvas

    # Build index: for each class, which dataset indices belong to it?
    import numpy as _np
    all_labels = _np.array(mixed_dataset.labels)

    fig, axes = plt.subplots(3, n_cls, figsize=(2.2 * n_cls, 7))
    fig.suptitle(
        "Cluster Sprites and Average Images  (Chen et al. 2018, Sec. 5.1)\n"
        "Row 1: Cluster A avg (clean)  |  "
        "Row 2: Cluster B avg (suspect)  |  "
        "Row 3: Cluster B sprite mosaic",
        fontsize=11, fontweight='bold',
    )

    for col, cls in enumerate(classes):
        cr        = cluster_map[cls]
        cls_idxs  = _np.where(all_labels == cls)[0]

        # Guard: cluster label array must match class sample count
        if len(cr.km_labels) != len(cls_idxs):
            for row in range(3):
                axes[row, col].axis('off')
            continue

        imgs_a = [mixed_dataset[int(i)][0]
                  for i, km in zip(cls_idxs, cr.km_labels)
                  if km == cr.larger_cluster]
        imgs_b = [mixed_dataset[int(i)][0]
                  for i, km in zip(cls_idxs, cr.km_labels)
                  if km == cr.smaller_cluster]

        cmap = 'gray' if dataset_info.n_channels == 1 else None

        # Row 1 — Cluster A average
        ax = axes[0, col]
        if imgs_a:
            ax.imshow(avg_img(imgs_a), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(
            f"class {cls}\nA avg\n({len(imgs_a)})",
            fontsize=7, fontweight='bold'
        )
        ax.axis('off')

        # Row 2 — Cluster B average
        ax = axes[1, col]
        if imgs_b:
            ax.imshow(avg_img(imgs_b), cmap=cmap, vmin=0, vmax=1)
        ax.set_title(
            f"B avg\n(suspect)\n({len(imgs_b)})",
            fontsize=7,
        )
        ax.axis('off')

        # Row 3 — Cluster B sprite
        ax = axes[2, col]
        if imgs_b:
            sprite      = make_sprite(imgs_b)
            sprite_cmap = None if dataset_info.n_channels == 3 else cmap
            ax.imshow(sprite, cmap=sprite_cmap, vmin=0, vmax=1,
                      interpolation='nearest')
        ax.set_title("B sprite", fontsize=7)
        ax.axis('off')

    # Row labels on left
    axes[0, 0].set_ylabel("A avg\n(clean)",   fontsize=9, fontweight='bold')
    axes[1, 0].set_ylabel("B avg\n(suspect)", fontsize=9, fontweight='bold')
    axes[2, 0].set_ylabel("B sprite",         fontsize=9, fontweight='bold')

    plt.tight_layout()

    if save:
        os.makedirs(results_dir, exist_ok=True)
        path = os.path.join(results_dir, "cluster_sprites.png")
        plt.savefig(path, dpi=130, bbox_inches='tight')
        print(f"Saved → {path}")

    if show:
        plt.show()
    
    plt.close(fig)