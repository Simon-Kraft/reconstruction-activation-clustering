# Backdoor Detection via Activation Clustering with Gradient-Inverted Poison

A reimplementation and extension of **Chen et al. (2018) "Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering"**, replacing direct pixel poisoning with **Geiping et al. (2020) gradient inversion** as the poison source.

---

## Research Question

Chen et al. (2018) assume that poisoned training samples are clean original images with a trigger patch stamped directly onto the pixels. This project tests what happens when that assumption is violated:

> *Does Activation Clustering still detect backdoors when the attacker reconstructs source images via gradient inversion (Geiping et al., 2020) rather than using clean originals?*

This is a more realistic threat model — in federated learning, an attacker intercepts gradients from a victim's update and reconstructs their training images before injecting the trigger, without ever having direct access to the original data.

---

## Key Results

| Poison Rate | Paper AC F1 | This Project AC F1 | Paper Raw F1 | This Project Raw F1 |
|---|---|---|---|---|
| 10% | ~99.96% | TBD | ~15.8% | TBD |
| 15% | ~99.9% | ~0.60–0.71 | ~15.8% | ~0.40–0.60 |
| 33% | ~99.9% | **1.0000** | ~15.8% | ~0.8224 |

**Main finding:** At high poison rates (33%), AC still detects all 10 poisoned classes perfectly even with gradient-inverted reconstructions. At lower rates (15%), detection degrades significantly and becomes class-pair dependent. The 2D silhouette score of the activation clusters is a reliable predictor of whether detection will succeed.

---

## Method Overview

### Threat Model

```
Standard (Chen et al.):       Attacker has clean source images
                               → stamps trigger directly → poisons dataset

This project (extended):      Attacker intercepts gradients from federated update
                               → Geiping reconstruction → stamps trigger → poisons dataset
```

### Rotating Poison Scheme

Following the paper exactly, all 10 MNIST classes are poisoned simultaneously:

```
class 0 → reconstructed "0"s with trigger, labelled as class 1
class 1 → reconstructed "1"s with trigger, labelled as class 2
...
class 9 → reconstructed "9"s with trigger, labelled as class 0
```

### Full Pipeline

```
Step 1  Load dataset (MNIST)
Step 2  Build poisoned dataset
          For each pair (lm → lm+1 mod 10):
            - Select p% of class lm samples
            - Intercept gradients from untrained CNN for each image
            - Reconstruct via Geiping cosine gradient inversion (75 iterations)
            - Stamp 3×3 trigger patch at bottom-right corner
            - Append as class (lm+1) samples
          Cache result to disk for reuse
Step 3  Train backdoor model (PaperCNN, 10 epochs) on poisoned dataset
Step 4  Verify backdoor (clean accuracy + attack success rate on 0→1)
Step 5  Extract activations
          - fc1 activations (128D) per training sample, grouped by class
          - Raw pixel values (784D) per training sample (baseline)
Step 6  Cluster (per method)
          - Normalise features (zero mean, unit std)
          - FastICA → 10D (falls back to PCA if ICA fails to converge)
          - K-means (k=2)
          - Smaller cluster = predicted poisoned
Step 7  Analyse clusters
          - Silhouette score in ICA space and 2D PCA projection
          - Size ratio (smaller cluster / total class size)
          - Flag class as poisoned if both thresholds exceeded
Step 8  Evaluate detection
          - F1 score per class (clustering quality, not just flagging)
          - AC vs raw clustering comparison table
Step 9  Visualise
          - Activation scatter: ground truth vs k-means (all 10 classes)
          - Silhouette bar chart
          - Reconstruction grid: original vs reconstructed+trigger per pair
          - Cluster sprites: average image and mosaic per cluster
```

### Reconstruction via Geiping et al. (2020)

The gradient inversion minimises the cosine distance between gradients of a dummy image and the intercepted target gradients:

```
loss = 1 - cos(∇dummy, ∇target) + λ_TV · TV(dummy)
```

Key properties:
- Uses magnitude-oblivious cosine loss (works on both trained and untrained models)
- Signed Adam updates for stable optimisation on ReLU networks
- Total variation regularisation encourages natural-looking images
- With untrained model: reconstructions recover shape but not fine detail
- With pretrained model: reconstructions are nearly indistinguishable from originals

---

## Project Structure

```
reconstruction-activation-clustering/
│
├── config.py                    ← All hyperparameters — edit this to run experiments
├── pipeline.py                  ← End-to-end orchestration (9 steps)
├── evaluate.py                  ← F1/accuracy metrics and comparison table
│
├── data/
│   ├── __init__.py
│   ├── loader.py                ← MNIST/CIFAR loading, normalisation stats
│   ├── trigger.py               ← Trigger patch config, auto-scales to image size
│   ├── reconstruction.py        ← Geiping gradient inversion (ReconConfig, reconstruct)
│   └── builder.py               ← Rotating poison builder, MixedDataset, caching
│
├── models/
│   ├── __init__.py
│   ├── cnn.py                   ← PaperCNN matching Chen et al. architecture
│   └── train.py                 ← Training loop, evaluation, ASR computation
│
├── activation_clustering/
│   ├── __init__.py
│   ├── extractor.py             ← fc1 hook extraction + raw pixel baseline
│   ├── clustering.py            ← ICA/PCA reduction + k-means (ClusterResult)
│   └── analyzer.py              ← Silhouette, size ratio, flagging (AnalysisResult)
│
├── visualization/
│   ├── __init__.py
│   └── plots.py                 ← All figures: scatter, silhouette, sprites, grid
│
├── datasets/                    ← Raw MNIST/CIFAR downloads (auto-created)
├── cache/                       ← Reconstructed poisoned datasets (.pt files)
├── checkpoints/                 ← Trained model weights (.pt files)
└── results/                     ← Plots and metrics per experiment
    └── MNIST_rotating_r0.33_sub0.2_noise0.0_pre0/
        ├── activation_scatter_all_classes.png
        ├── silhouette_scores.png
        ├── reconstructed_samples.png
        ├── cluster_sprites.png
        ├── ac_detection_results.json
        └── raw_detection_results.json
```

### Model Architecture (PaperCNN)

Matches Chen et al. (2018) exactly:

```
Conv2d(1, 32, 3, pad=1) → ReLU → MaxPool2d(2)
Conv2d(32, 64, 3, pad=1) → ReLU → MaxPool2d(2)
Flatten
Linear(3136, 128) → ReLU    ← fc1: AC extracts activations here
Linear(128, 10)
```

---

## Installation

```bash
git clone https://github.com/yourusername/reconstruction-activation-clustering
cd reconstruction-activation-clustering

# Create conda environment
conda create -n recon-ac python=3.11
conda activate recon-ac

# Install dependencies
pip install torch torchvision numpy scikit-learn matplotlib pillow tqdm
```

---

## Running Experiments

### Basic Run

```bash
python pipeline.py
```

MNIST downloads automatically to `datasets/` on first run. The poisoned dataset is built and cached to `cache/` — subsequent runs with the same config skip reconstruction entirely and load from cache.

### Changing Experiments

Edit `config.py` — the experiment ID is automatically derived from all parameters, so results never overwrite each other:

```python
# config.py — key parameters to change

POISON_CFG = PoisonConfig(
    poison_rate     = 0.15,    # 0.10, 0.15, or 0.33 — matches paper Table 1
    subsample_rate  = 0.2,     # 0.2 recommended — 20% of MNIST (~12k samples)
    pretrain_epochs = 0,       # 0 = untrained model, 5 = pretrained reconstruction
    noise_std       = 0.0,     # 0.0 = clean, 0.05/0.1 = noisy gradients
    dlg_iterations  = 75,      # reconstruction steps (75 is sufficient)
)

AC_METHOD = 'ica'              # 'ica' (paper default) or 'pca_2d'
```

**Always delete the cached dataset when changing any poison parameter:**

```bash
rm cache/mixed_MNIST_rotating_r<old_params>.pt
python pipeline.py
```

### Running All Three Poison Rates

```bash
# In config.py: poison_rate=0.10, subsample_rate=0.2
python pipeline.py

# In config.py: poison_rate=0.15, subsample_rate=0.2
python pipeline.py

# In config.py: poison_rate=0.33, subsample_rate=0.2
python pipeline.py
```

### Ablation: Noisy Gradients

```bash
# In config.py: noise_std=0.05
# Delete cache, run
python pipeline.py
```

### Ablation: Pretrained Reconstruction Model

```bash
# In config.py: pretrain_epochs=5
# Delete cache, run — this takes longer as the model is pretrained first
python pipeline.py
```

---

## Output

Each run saves results to `results/MNIST_rotating_r{rate}_sub{sub}_noise{noise}_pre{pretrain}/`:

| File | Description |
|---|---|
| `activation_scatter_all_classes.png` | 2-row grid: ground truth (top) and k-means clusters (bottom) for all 10 classes projected to 2D PCA |
| `silhouette_scores.png` | Bar chart of silhouette scores per class with detection threshold lines |
| `reconstructed_samples.png` | 4 examples per source→target pair: original (top) vs reconstructed+trigger (bottom) |
| `cluster_sprites.png` | Average image and thumbnail mosaic per cluster per class — human verification of detection |
| `ac_detection_results.json` | Per-class and overall F1/accuracy/precision/recall for AC method |
| `raw_detection_results.json` | Same for raw pixel clustering baseline |

### Reading the Results Table

```
class   method     sil    AC F1   Raw F1   flagged  correct
    0      ica   0.089   1.0000   0.7574       YES        ✅
```

- **method** — ICA or PCA (ICA falls back to PCA if it fails to converge)
- **sil** — silhouette score of the k-means split in the clustering space
- **AC F1** — F1 score for fc1 activation clustering (smaller cluster = poisoned)
- **Raw F1** — F1 score for raw pixel clustering (baseline)
- **flagged** — whether the class exceeded the silhouette+size threshold
- **correct** — whether the flagging decision was correct

---

## Differences from Chen et al. (2018)

| Aspect | Paper | This Project |
|---|---|---|
| Poison source | Clean original images + trigger | Geiping-reconstructed images + trigger |
| Poison scheme | Rotating lm→(lm+1)%10 | ✅ Same |
| Architecture | PaperCNN | ✅ Same |
| AC layer | fc1 | ✅ Same |
| Dimensionality reduction | ICA → 10D | ✅ Same |
| Detection metric | Per-class accuracy and F1 | ✅ Same |
| Raw clustering baseline | ✅ | ✅ Added |
| Silhouette analysis | ✅ | ✅ Extended (10D + 2D) |
| Exclusionary reclassification | ✅ | Not implemented |
| LISA traffic signs | ✅ | Not implemented |
| Rotten Tomatoes text | ✅ | Not implemented |

---

## References

- Chen, B., Carvalho, W., Baracaldo, N., Ludwig, H., Edwards, B., Lee, T., Molloy, I., & Srivastava, B. (2018). *Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering.* AAAI Workshop on Artificial Intelligence Safety.

- Geiping, J., Bauermeister, H., Dröge, H., & Moeller, M. (2020). *Inverting Gradients — How easy is it to break privacy in federated learning?* NeurIPS 2020.

- Gu, T., Dolan-Gavitt, B., & Garg, S. (2017). *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.* arXiv:1708.06733.