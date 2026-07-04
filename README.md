# Reconstruction-Based Activation Clustering Backdoor Detection

> Probing whether gradient-inverted reconstructions fool activation clustering — CPSC 461/661 · UNBC Winter 2026

Authors: Nazanin Parvizi, Sukirat Singh Dhillon, Simon Kraft

---

## Overview

Evaluates Activation Clustering (AC) backdoor detection when poisoned
training samples are reconstructed from intercepted gradients via the
Geiping gradient-inversion attack.


## Structure
```
reconstruction-activation-clustering/
│
├── pipeline.py            # Main end-to-end pipeline (9-step orchestration)
├── ac_sweep.py            # AC sweep over n_components (steps 6–9, called by pipeline)
├── config.py              # All hyperparameters, paths, and experiment config
├── evaluate.py            # Detection evaluation: F1, accuracy, result saving
│
├── data/
│   ├── loader.py          # Dataset loading for MNIST, FashionMNIST, CIFAR10
│   ├── builder.py         # Builds rotating poisoned dataset (MixedDataset)
│   ├── reconstruction.py  # Geiping gradient inversion implementation
│   └── trigger.py         # Trigger config and injection (3x3 patch)
│
├── models/
│   ├── cnn.py             # PaperCNN architecture with forward hooks for AC
│   └── train.py           # Training loop, evaluation, ASR, checkpointing
│
├── clustering/
│   ├── extractor.py       # Extracts fc1 activations and raw pixel features
│   ├── clustering.py      # ICA/PCA dimensionality reduction + k-means
│   └── analyzer.py        # Silhouette score, size ratio, poison flagging
│
├── visualization/
│   ├── plots.py           # Activation scatter, silhouette bars, cluster sprites
│   └── visualize_3d.py    # 3D PCA activation visualisation
│
├── scripts/
│   ├── run_mnist.sh                        # Core MNIST experiments
│   ├── run_fashionmnist.sh                 # Core FashionMNIST experiments
│   ├── run_noise_ablation.sh               # Noise + pretraining ablation at p=15%
│   ├── run_reconstruction_comparison.sh    # Geiping vs DLG comparison
│   ├── plot_ablation.py                    # 2×2 noise/pretrain ablation figure
│   ├── plot_ablations.py                   # Alternative ablation plots
│   ├── generate_latex_tables.py            # LaTeX result tables from summary logs
│   ├── generate_ablation_tables.py         # LaTeX ablation tables
│   ├── generate_silhouette_tables.py       # LaTeX silhouette-score tables
│   ├── generate_silhouette_ablation_tables.py  # Silhouette ablation tables
│   └── verify_architecture.py             # Verify CNN parameter counts vs paper
│
├── gfx/                   # Generated figures for the report
├── datasets/              # Generated and downloaded datasets (auto-created)
├── checkpoints/           # Trained model checkpoints (auto-created)
├── logs/                  # Experiment logs and result CSVs (auto-created)
└── results/               # Per-experiment output figures and JSON (auto-created)
```

## Installation

Python 3.11 is required. Install dependencies with:

```bash
pip install torch torchvision scikit-learn matplotlib numpy tqdm
```

For GPU acceleration, install PyTorch with CUDA support following the
[official instructions](https://pytorch.org/get-started/locally/) for your platform.

## Running

**Single experiment:**
```bash
python pipeline.py --dataset MNIST --poison_rate 0.15
```

**Full experiment suites:**
```bash
chmod +x scripts/*.sh
./scripts/run_mnist.sh
./scripts/run_fashionmnist.sh
./scripts/run_noise_ablation.sh
./scripts/run_reconstruction_comparison.sh
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `MNIST` | `MNIST`, `FashionMNIST`, or `CIFAR10` |
| `--poison_rate` | `0.15` | Fraction of each class to poison |
| `--subsample_rate` | `0.25` | Fraction of training set to use |
| `--reconstruction_method` | `geiping` | `geiping` = cosine inversion, `dlg` = L2 inversion, `badnets` = no reconstruction |
| `--noise_std` | `0.0` | Gaussian noise added to intercepted gradients |
| `--pretrain_epochs` | `0` | Epochs to pre-train before gradient inversion |
| `--layer` | `fc1` | Activation layer(s) for AC, e.g. `conv1,fc1` |
| `--ac_n_components` | `2` | ICA/PCA components to evaluate, e.g. `2,4,6,10` |
| `--seed` | `42` | Random seed |
| `--replace_originals` | off | Replace reconstructed source images instead of appending |
| `--no_plots` | off | Suppress figure generation |

Datasets, poisoned caches, and model checkpoints are saved automatically
and reused on subsequent runs to avoid redundant computation.
