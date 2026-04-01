# Backdoor Detection via Activation Clustering (AC)

A PyTorch implementation of the backdoor detection method from:

> **Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering**  
> Chen et al., 2018 — [arXiv:1811.03728](https://arxiv.org/abs/1811.03728)

Combined with a **gradient-based image reconstruction attack (DLG)** to simulate a realistic poisoning scenario without access to the raw training data.

---

## The Core Idea

Modern ML pipelines often rely on data from untrusted sources — crowdsourcing platforms, third-party datasets, federated learning clients. An adversary with access to any part of this pipeline can insert **backdoored samples**: images that look normal to a human but cause the model to misbehave whenever a specific trigger (e.g. a small pixel patch) is present at inference time.

The key challenge: the backdoored model **performs perfectly on clean test data**, so standard accuracy metrics cannot detect the attack. Something deeper is needed.

---

## Our Pipeline

```
Full MNIST training set (60,000 images)
         │
         ▼
  Select N samples from non-target classes
         │
         ▼  [poison.py]
  Reconstruct each via DLG attack
  (simulate gradient leakage from a federated client)
         │
         ▼
  Stamp trigger patch + flip label → target class
         │
         ▼
  Replace originals in training set with poisoned reconstructions
         │
         ▼  [train.py]
  Train model on full mixed dataset (~60k samples)
  → model learns normally AND learns the backdoor
         │
         ▼  [extract.py]
  Extract activations from every layer
  for all TARGET_CLASS samples only
         │
         ▼  [detect.py]
  Run Activation Clustering (PCA → KMeans → metrics)
         │
         ▼  [plot.py]
  Visualise separation + report silhouette scores
```

---

## Repository Structure

```
backdoor-ac/
│
├── config.py       — All hyperparameters in one place
├── model.py        — CNN architectures (LargeCNN, MidCNN, SmallCNN)
├── dlg.py          — Deep Leakage from Gradients reconstruction attack
├── poison.py       — Dataset builder: DLG + trigger injection
├── train.py        — Training and evaluation loop
├── extract.py      — Activation extraction from hooked layers
├── detect.py       — Activation Clustering detection and metrics
├── pipeline.py     — Orchestrates the full end-to-end run
├── plot.py         — Visualisation of results
│
├── data/           — MNIST download + cached poisoned dataset
├── checkpoints/    — Saved model weights
└── results/        — Saved metrics and plots
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install torch torchvision scikit-learn pandas tqdm matplotlib
```

### 2. Run the pipeline
```bash
python pipeline.py
```

### 3. Plot the results
```bash
python plot.py
```

The pipeline caches the poisoned dataset to `data/poisoned_recon_dataset.pt` after the first run. Subsequent runs skip the slow DLG reconstruction step and load from disk directly. To force a rebuild (e.g. after changing model or noise settings), delete the cache:

```bash
rm data/poisoned_recon_dataset.pt
```

---

## Configuration

Everything is controlled from `config.py`. The most important knobs:

| Parameter | Default | Description |
|---|---|---|
| `MODEL` | `'LargeCNN'` | Architecture to use (`LargeCNN`, `MidCNN`, `SmallCNN`) |
| `N_POISON` | `20` | How many samples to reconstruct and poison |
| `TARGET_CLASS` | `0` | Which class receives the backdoor |
| `RECON_PRETRAIN_EPOCHS` | `0` | Epochs to pretrain reconstruction model before DLG |
| `DLG_NOISE_STD` | `0.0` | Gaussian noise added to intercepted gradients |
| `DLG_ITERATIONS` | `300` | L-BFGS steps per reconstruction |
| `TRAIN_EPOCHS` | `10` | Epochs to train the backdoor model |

---

## How the Detection Works (Chen et al., 2018)

### Intuition

Even though poisoned and clean samples share the same predicted label (e.g. class 0), the **reason** the model assigns that label is completely different:

- A **clean "0"** digit → the model activates because it recognises the shape of a zero.
- A **poisoned "9" with a trigger** → the model activates because it sees the trigger patch. The underlying image still looks like a 9 to the network's early layers.

This difference in mechanism is visible in the **activations of the last hidden layer** — the internal representation the model built to make its decision. When projected to 2D with PCA, clean and poisoned samples separate into two distinct clusters even though they carry the same label.

### Algorithm

```
For each class label c:
  1. Extract activations of the last hidden layer for all samples labelled c
  2. Reduce dimensionality (ICA in the paper, PCA in our implementation)
  3. Cluster with KMeans(k=2)
  4. Analyse clusters to determine if one is poisoned:
       - Silhouette score:          high → two real clusters exist → likely poisoned
       - Relative cluster size:     small cluster ≈ poison_rate → likely poisoned
       - Exclusionary reclassification: retrain without cluster, classify removed
                                        samples → poisoned ones flip to source class
```

### Metrics we compute (detect.py)

| Metric | What it measures | Sign of poison |
|---|---|---|
| **Silhouette score** | How well-separated the two clusters are | > 0.10–0.15 (paper threshold) |
| **Cluster purity** | Fraction of samples correctly grouped by majority vote | Close to 1.0 |
| **LDA score** | Linear separability between clean and poisoned in 2D | Close to 1.0 |
| **Composite** | Equal-weight average of the three above | Higher = more detectable |

---

## Experiments & Ideas

All experiments are controlled purely through `config.py`. Delete `data/poisoned_recon_dataset.pt` and `results/metrics.pt` between experiments to force a fresh run.

### Experiment A — Effect of reconstruction model pretraining

**Question:** Does the quality of the reconstructed images affect how detectable the backdoor is?

A randomly initialised model produces gradients that carry very little semantic information. DLG reconstructions from these gradients are noisy and low-quality. If the reconstruction model has been pretrained for a few epochs, the gradients are more structured and the reconstructions are more faithful to the originals — making the injected backdoor potentially more realistic and harder or easier to detect.

```python
# config.py
RECON_PRETRAIN_EPOCHS = 0    # baseline: random model
RECON_PRETRAIN_EPOCHS = 2    # lightly pretrained
RECON_PRETRAIN_EPOCHS = 5    # moderately pretrained
RECON_PRETRAIN_EPOCHS = 10   # well pretrained
```

**Expected observation:** Higher pretraining → more faithful reconstructions → poisoned images more closely resemble the source class → potentially clearer cluster separation in activation space.

---

### Experiment B — Effect of gradient noise on reconstruction quality

**Question:** If we add noise to the intercepted gradients (simulating a privacy defence like differential privacy), can we degrade the quality of the reconstruction enough to prevent the backdoor from being learned?

```python
# config.py
DLG_NOISE_STD = 0.0     # no noise (baseline)
DLG_NOISE_STD = 0.01    # light noise
DLG_NOISE_STD = 0.1     # moderate noise
DLG_NOISE_STD = 0.5     # heavy noise
```

**Expected observation:** Higher noise → lower reconstruction quality → poisoned images look more random → the model may not learn the backdoor reliably → weaker cluster separation in activation space → lower silhouette score.

---

### Experiment C — Different model architectures

**Question:** Does the architecture of the model affect how clearly the backdoor clusters in activation space?

```python
# config.py
MODEL = 'LargeCNN'   # deeper, more capacity
MODEL = 'MidCNN'     # medium, sigmoid activations
MODEL = 'SmallCNN'   # shallow, fewer parameters
```

**Expected observation:** Larger models with more capacity tend to learn more disentangled representations, making clean/poisoned clusters easier to separate. Sigmoid activations (MidCNN) may compress activations differently than ReLU.

---

## Ideas for Further Work

### 1. Higher-dimensional detection without PCA

The paper reduces to 10 ICA components before clustering, which discards information. In high-dimensional activation spaces this is necessary to avoid the **curse of dimensionality** (distance metrics become unreliable as dimensions grow). However, several alternatives avoid this information loss:

- **UMAP** instead of PCA — preserves local structure better and often separates clusters more cleanly in 2D without reducing to 10 components first.
- **Isolation Forest** — an anomaly detection method that works well in high dimensions by isolating outliers through random splits. Poisoned samples, being atypical members of their class, should be easier to isolate.
- **One-Class SVM** — train on clean activations only (if a small trusted set is available), then flag everything outside the decision boundary.
- **Mahalanobis distance** — measures how far each sample's activation vector is from the class mean, accounting for covariance. Poisoned samples tend to have high Mahalanobis distance from the clean class distribution. This has been shown to work well for OOD detection in neural networks (Lee et al., 2018).

### 2. Multi-layer detection

Currently we run AC independently per layer and pick the best one. A natural extension is to **combine information across layers** to make the detection decision more robust:

- **Concatenate activations** from multiple layers into a single feature vector per sample, then run PCA + KMeans on the combined representation. This gives the detector access to both low-level (early layers) and high-level (late layers) information simultaneously.
- **Ensemble voting** — run AC on each layer independently and take a majority vote across layers. If 4 out of 6 layers flag a sample as poisoned, classify it as poisoned. This reduces the chance of a false positive from a single noisy layer.
- **Weighted combination** — weight each layer's vote by its silhouette score. Layers that produce cleaner cluster separation get more influence over the final decision.
- **Layer-wise anomaly scores** — instead of hard cluster assignments, compute a continuous anomaly score per sample per layer (e.g. distance to the clean cluster centroid), then aggregate scores across layers with a learned or heuristic weighting.

### 3. Why knowing which samples are backdoored matters

Once you have identified the poisoned samples, two remediation strategies become available:

**Remove and retrain** — delete the identified poisoned samples entirely and retrain from scratch on the remaining clean data. Simple and reliable, but expensive if the dataset is large.

**Relabel and fine-tune** (recommended by Chen et al.) — relabel the poisoned samples back to their true source class (e.g. from "0" back to "9") and continue training for a small number of additional epochs until the model unlearns the backdoor. The paper found this converged in **14 epochs** compared to **80 epochs** for full retraining — a 5× speedup. The model's accuracy on clean data is preserved throughout.

Neither strategy is possible without knowing **which specific samples** are poisoned. Without this information, the entire dataset must be discarded, which in real-world settings (crowdsourced data, federated learning, third-party model catalogues) is often impractical or impossible.

---

## References

- Chen et al. (2018). *Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering.* [arXiv:1811.03728](https://arxiv.org/abs/1811.03728)
- Zhu et al. (2019). *Deep Leakage from Gradients.* [arXiv:1906.08935](https://arxiv.org/abs/1906.08935)
- Gu et al. (2017). *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain.* [arXiv:1708.06733](https://arxiv.org/abs/1708.06733)
- Lee et al. (2018). *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.* [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)