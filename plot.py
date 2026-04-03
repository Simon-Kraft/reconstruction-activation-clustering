"""
plot.py — Simple activation clustering visualisation for the target class.

Loads results/metrics.pt saved by pipeline.py and shows a PCA scatter
of clean vs poisoned activations for every layer, plus a silhouette
score bar chart so you can see which layer separates best.

Run with:
    python plot.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import config as C
from poison import MixedDataset

# ---------------------------------------------------------------------------
# ✏️  Change this to show more or fewer images in Figure 3
# ---------------------------------------------------------------------------
N_SHOW       = 30   # total number of poisoned samples to display
COLS_PER_ROW = 10   # how many images per row

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
results    = torch.load(C.RESULTS_METRICS_PATH, weights_only=False)
df         = pd.DataFrame(results["metrics"])
extraction = results["extraction"]

feats = extraction["feats"]   # dict: layer_name → (N, D) array
flags = extraction["flags"]   # (N,) bool array — True = poisoned

# Subsample clean points down to match the number of poisoned ones
# so the scatter is balanced and easy to read
rng           = np.random.default_rng(C.SEED)
n_poison      = flags.sum()
clean_idxs    = np.where(~flags)[0]
sampled_clean = rng.choice(clean_idxs, size=n_poison, replace=False)
keep          = np.sort(np.concatenate([sampled_clean, np.where(flags)[0]]))
feats         = {layer: arr[keep] for layer, arr in feats.items()}
flags         = flags[keep]
print(f"Plotting {n_poison} poisoned + {n_poison} clean "
      f"(subsampled from {len(clean_idxs)} clean total)")

layers   = list(feats.keys())
n_layers = len(layers)

# ---------------------------------------------------------------------------
# Figure 1 — PCA scatter for every layer
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, n_layers, figsize=(4 * n_layers, 8))
fig.suptitle(
    f"Activation Clustering — Target class {C.TARGET_CLASS}\n"
    f"Blue = clean  |  Red = poisoned  |  ■ = KMeans cluster boundary",
    fontsize=13, fontweight="bold"
)

for col, layer in enumerate(layers):
    feat = feats[layer]
    fn   = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)
    p2   = PCA(n_components=2, random_state=C.SEED).fit_transform(fn)
    km   = KMeans(n_clusters=2, random_state=C.SEED, n_init=10).fit(p2)

    # --- Top row: raw scatter coloured by ground truth ---
    ax = axes[0, col]
    ax.scatter(p2[~flags, 0], p2[~flags, 1],
               c="#2980b9", s=18, alpha=0.6, label="Clean")
    ax.scatter(p2[flags,  0], p2[flags,  1],
               c="#e74c3c", s=18, alpha=0.8, label="Poisoned")
    ax.set_title(layer, fontweight="bold")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.grid(alpha=0.2)
    if col == 0:
        ax.legend(fontsize=8)

    # --- Bottom row: same scatter + KMeans decision boundary ---
    ax2 = axes[1, col]
    x0, x1 = p2[:, 0].min() - 1, p2[:, 0].max() + 1
    y0, y1 = p2[:, 1].min() - 1, p2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x0, x1, 200),
                         np.linspace(y0, y1, 200))
    Z = km.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax2.contourf(xx, yy, Z, alpha=0.10, colors=["#3498db", "#e74c3c"])
    ax2.contour(xx, yy, Z, colors=["gray"], linewidths=0.8,
                linestyles="--", alpha=0.5)
    ax2.scatter(p2[~flags, 0], p2[~flags, 1],
                c="#2980b9", s=18, alpha=0.6)
    ax2.scatter(p2[flags,  0], p2[flags,  1],
                c="#e74c3c", s=18, alpha=0.8)

    sil = df.loc[layer, "silhouette"]
    ax2.set_title(f"+ KMeans  (sil={sil:.3f})", fontsize=9)
    ax2.set_xlabel("PC 1"); ax2.set_ylabel("PC 2")
    ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(C.RESULTS_DIR + "activation_clustering.png", dpi=130, bbox_inches="tight")
plt.show()
print(f"Saved → {C.RESULTS_DIR}activation_clustering.png")

# ---------------------------------------------------------------------------
# Figure 2 — Silhouette score bar chart across layers
# ---------------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(8, 4))
colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df["silhouette"]]
bars   = ax.bar(df.index, df["silhouette"], color=colors,
                edgecolor="white", alpha=0.85)
ax.axhline(0,    color="black",  linewidth=0.8)
ax.axhline(0.10, color="orange", linewidth=1, linestyle="--",
           label="Paper threshold (0.10)")
ax.axhline(0.15, color="red",    linewidth=1, linestyle="--",
           label="Paper threshold (0.15)")
for bar, val in zip(bars, df["silhouette"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            val + 0.005, f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_title(f"Silhouette Score per Layer — Target class {C.TARGET_CLASS}",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Silhouette score")
ax.set_xlabel("Layer")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(C.RESULTS_DIR + "silhouette_scores.png", dpi=130, bbox_inches="tight")
plt.show()
print(f"Saved → {C.RESULTS_DIR}silhouette_scores.png")

# ---------------------------------------------------------------------------
# Figure 3 — Original vs Reconstructed+Triggered image grid
# ---------------------------------------------------------------------------
mixed_dataset = MixedDataset.load(C.RECON_DATASET_PATH)

# Collect (original, reconstructed) pairs for poisoned samples
pairs = [
    (mixed_dataset.orig_data[i], mixed_dataset.data[i])
    for i in range(len(mixed_dataset))
    if mixed_dataset.is_poisoned[i]
]

n_show = min(N_SHOW, len(pairs))
n_rows = -(-n_show // COLS_PER_ROW)   # ceiling division → number of image rows
# Figure has 2 subrow per image row (original + reconstructed)
fig_rows = n_rows * 2

MEAN, STD = C.MEAN, C.STD
def to_display(t):
    """Unnormalise tensor (C,H,W) → numpy (H,W) for imshow."""
    return (t * STD + MEAN).squeeze().numpy().clip(0, 1)

fig3, axes3 = plt.subplots(
    fig_rows, COLS_PER_ROW,
    figsize=(COLS_PER_ROW * 1.6, fig_rows * 1.6)
)
# Ensure axes3 is always 2D
if fig_rows == 1:
    axes3 = axes3[np.newaxis, :]
fig3.suptitle(
    f"Original (odd rows) vs Reconstructed + Trigger (even rows)\n"
    f"Showing {n_show} poisoned samples  ({COLS_PER_ROW} per row)",
    fontsize=12, fontweight="bold"
)

for idx, (orig, recon) in enumerate(pairs[:n_show]):
    img_row  = idx // COLS_PER_ROW   # which group of rows we're in
    col      = idx  % COLS_PER_ROW   # which column

    orig_ax  = axes3[img_row * 2,     col]
    recon_ax = axes3[img_row * 2 + 1, col]

    orig_ax.imshow(to_display(orig),  cmap="gray", vmin=0, vmax=1)
    orig_ax.axis("off")
    recon_ax.imshow(to_display(recon), cmap="gray", vmin=0, vmax=1)
    recon_ax.axis("off")

    # Label the first column of each pair of rows
    if col == 0:
        orig_ax.set_ylabel("Original",      fontsize=8, labelpad=2)
        recon_ax.set_ylabel("Recon+Trigger", fontsize=8, labelpad=2)

# Hide any unused axes in the last row
for idx in range(n_show, n_rows * COLS_PER_ROW):
    img_row = idx // COLS_PER_ROW
    col     = idx  % COLS_PER_ROW
    axes3[img_row * 2,     col].axis("off")
    axes3[img_row * 2 + 1, col].axis("off")

plt.tight_layout()
plt.savefig(C.RESULTS_DIR + "reconstructed_samples.png", dpi=130, bbox_inches="tight")
plt.show()
print(f"Saved → {C.RESULTS_DIR}reconstructed_samples.png")