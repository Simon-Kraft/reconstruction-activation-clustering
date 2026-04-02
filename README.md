# Backdoor Detection via Activation Clustering
## CPSC 461 Final Project

This notebook investigates whether **backdoor attacks** are more detectable in neural networks when the poisoned training data comes from **DLG-reconstructed images** versus **original images**.

---

## Research Question

> Does training a model on DLG-reconstructed poisoned data make the backdoor *more detectable* via activation clustering compared to training on standard poisoned data?

---

## Notebook Structure: `recon_vs_original.ipynb`

| Cell # | Type | Description |
|--------|------|-------------|
| 0 | Markdown | Title header |
| 1 | Code | **Imports & setup** — PyTorch, torchvision, scikit-learn (PCA, t-SNE, LDA, KMeans), pandas, matplotlib. Sets `SEED=42`, detects CUDA/CPU. |
| 2 | Markdown | Step 1 header |
| 3 | Code | MNIST normalization constants (`MEAN=0.1307, STD=0.3081`) |
| 4 | Code | **Load MNIST** — `transforms.Compose` with ToTensor + Normalize; creates `train_loader` (batch 64) and `test_loader` (batch 1000) |
| 5 | Markdown | Step 2 header |
| 6 | Code | **`InstrumentedCNN`** — 4 conv blocks + 3 FC layers (~1.77M params). Registers `forward_hook` on 10 layers (`conv1, bn1, conv2, bn2, conv3, bn3, conv4, bn4, fc1, fc2`) to capture activations automatically during every forward pass. Also defines `LAYER_META` dictionary with depth/channel/spatial metadata per layer. |
| 7 | Markdown | Step 3 header |
| 8 | Code | **`run_dlg_attack()`** — Deep Leakage from Gradients (DLG) attack. Given intercepted gradients from one training step, optimizes a dummy image via LBFGS (200 iterations) until its gradients match the real ones. Returns reconstructed image + predicted label. |
| 9 | Markdown | Step 4 header |
| 10 | Code | Extracts one test batch; sets up gradient computation for DLG demo |
| 11 | Code | **DLG demo + visualization** — Runs attack on one sample; plots Original \| Reconstructed \| Difference with MSE |
| 12 | Markdown | Step 5 header |
| 13 | Code | **Backdoor trigger config** — 3×3 pixel patch at position (24,24), intensity 2.8, target class 0, poison rate 10% |
| 14 | Code | **`PoisonedReconstructedDataset`** class — Builds a dataset by: (1) running DLG on N training samples, (2) poisoning 10% of non-target samples by stamping the trigger and flipping the label to class 0, (3) storing original images, reconstructed images, labels, and `is_poisoned` flags. Has `save()`/`load()` for disk persistence. |
| 15 | Code | Load or build the reconstructed poisoned dataset (500 samples, ~43 poisoned) |
| 16 | Code | `plot_dataset()` — Grid visualization of dataset samples with color-coded poison status |
| 17 | Code | Create `recon_poison_loader` (DataLoader, batch 16) |
| 18 | Markdown | Step 7 header |
| 19 | Code | **`train()`** — Standard training loop with Adam (lr=1e-3) + CosineAnnealingLR scheduler. Trains `backdoor_model` on reconstructed poisoned data for 10 epochs (~75% → 94% accuracy). |
| 20 | Markdown | Step 8 header |
| 21 | Code | **`extract_layer_features()`** — Runs all images through the model; collects hooked activations. Conv/BN layers → global average pooled to `(N, C)`. FC layers → raw `(N, D)`. Returns dict of `(N, D)` arrays per layer. |
| 22 | Code | Collect all 500 images + `is_poisoned` flags; call `extract_layer_features` on `backdoor_model`. Prints feature shapes per layer. |
| 23 | Markdown | Step 9 header + metric explanations |
| 24 | Code | **Layer significance scoring** — Defines `silhouette()`, `fisher_ratio()`, `kmeans_purity()`, `lda_accuracy()`. Runs all 4 on every layer; normalizes scores; computes composite score; identifies `BEST_LAYER` and `WORST_LAYER`. |
| 25 | Markdown | Step 10 header |
| 26 | Code | **Layer score bar/line plots** — 6-subplot figure showing composite + individual metric scores across all 10 layers, highlighting best/worst. |
| 27 | Markdown | Step 11 header |
| 28 | Code | **2D projection functions** — `project_2d()` (PCA / t-SNE / LDA) and `scatter_with_clusters()` (scatter + K-Means decision boundary background). Pre-computes projections for best, penultimate (`conv3`), and worst layers. |
| 29 | Code | **3×3 clustering grid** — Rows = 3 comparison layers, Cols = PCA / t-SNE / LDA. Shows clean (blue) vs. poisoned (red) with K-Means background shading. |
| 30 | Code | **`PoisonedDataset`** class — Standard poisoned dataset (same trigger, same rate, but using original MNIST images — no DLG). Baseline for comparison. Trains `standard_poison_model`. |
| 31 | Code | Train `standard_poison_model` on standard poisoned data (10 epochs, ~94% accuracy) |
| 32 | Code | Extract features from both datasets through their respective trained models |
| 33 | Code | **`compute_scores()`** — Runs all 4 metrics on both datasets; computes delta = recon − standard per layer and metric. Prints composite score comparison table. |
| 34 | Code | Side-by-side composite score bar chart (standard vs. reconstructed, all layers) |
| 35 | Code | Delta heatmap — color-coded per-metric improvement/degradation across layers |
| 36 | Code | Silhouette profile line plot + scatter comparison for one layer |
| 37 | Code | **Final comparison figure** (`comparison_clustering.png`) — 4×3 grid: Row 0 = silhouette + composite line plots; Rows 1–3 = standard scatter \| recon scatter \| delta bar for best, penultimate, worst layers. Saved at 130 DPI. |

---

## Key Findings (Original Notebook)

- Reconstructed poisoned data is **generally more detectable** (higher composite scores) than standard poisoned data across most layers.
- **Best layer for detection**: `fc1` (composite ≈ 0.75).
- **Exception — `fc2` layer**: Reconstructed poison shows a **negative delta**, meaning the standard poison is actually *more* detectable there.

---

## Changes Added: Improved Clustering for `fc2`

The following 5 cells were appended to the notebook (cells 38–42) to address the `fc2` weakness.

### Problem
At `fc2`, the reconstructed poison's activation clusters overlap more than the standard poison's — the opposite of what we want. This happens because `fc2` is only 128-D and PCA to 2-D discards a lot of discriminative structure.

### Strategy 1 — High-dim K-Means on `fc2` (Cell 39)
Instead of reducing to 2-D PCA before clustering, reduce to **20-D PCA** first, run K-Means in that space, then use **t-SNE only for visualization**. The clustering decision is made with 20× more information.

### Strategy 2 — Multi-layer Fusion (Cells 39–42)
Concatenate normalized features from **`conv4` + `fc1` + `fc2`** → a 512-D combined representation → **30-D PCA** → K-Means. Fusing neighboring layers gives the clustering algorithm context from earlier and later representations simultaneously, smoothing over the weakness of any single layer.

### New Cells Summary

| Cell # | Description |
|--------|-------------|
| 38 | Markdown explaining both strategies |
| 39 | Setup: `fuse_layers()`, `high_dim_cluster_project()`, compute all 4 projections (t-SNE ×4) |
| 40 | `score_approach()` — compute purity/silhouette/LDA for all 4 approach×dataset combos; print table |
| 41 | **`improved_clustering_fc2.png`** — 2×3 grid: Row 1 = fc2 high-dim approach (std \| recon \| delta bar), Row 2 = fused approach (std \| recon \| delta bar) |
| 42 | Summary table comparing original 2-D baseline vs. both improved strategies across purity/silhouette/LDA deltas |

### Output File
- `improved_clustering_fc2.png` — the main figure to show your professor/teammates

---

## How to Run

1. Open `recon_vs_original.ipynb` in Jupyter or VS Code
2. Run all cells top-to-bottom (the DLG dataset is cached to `data/poisoned_recon_dataset.pt` after the first build)
3. The new cells (38–42) depend on variables computed in cells 19–33 — run those first
4. t-SNE in cells 39 runs 4 times; expect ~2 minutes on CPU

## Dependencies
```
torch torchvision numpy matplotlib scikit-learn pandas tqdm
```
