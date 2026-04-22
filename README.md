# Reconstruction-Based Activation Clustering Backdoor Detection

CPSC 461/661 Applied Machine Learning — UNBC Winter 2026  
Authors: Nazanin Parvizi, Sukirat Singh Dhillon, Simon Kraft

## Overview

Evaluates Activation Clustering (AC) backdoor detection when poisoned
training samples are reconstructed from intercepted gradients via the
Geiping gradient-inversion attack.


## Structure
```
reconstruction-activation-clustering/
│
├── pipeline.py            # Main end-to-end pipeline (9-step orchestration)
├── config.py              # All hyperparameters, paths, and experiment config
├── evaluate.py            # Detection evaluation: F1, accuracy, result saving
│
├── data/
│   ├── loader.py          # Dataset loading for MNIST and FashionMNIST
│   ├── builder.py         # Builds rotating poisoned dataset (MixedDataset)
│   ├── reconstruction.py  # Geiping gradient inversion implementation
│   └── trigger.py         # Trigger config and injection (3x3 patch)
│
├── models/
│   ├── cnn.py             # PaperCNN architecture with forward hooks for AC
│   └── train.py           # Training loop, evaluation, ASR, checkpointing
│
├── activation_clustering/
│   ├── extractor.py       # Extracts fc1 activations and raw pixel features
│   ├── clustering.py      # ICA/PCA dimensionality reduction + k-means
│   └── analyzer.py        # Silhouette score, size ratio, poison flagging
│
├── visualization/
│   ├── plots.py           # Activation scatter, silhouette bars, cluster sprites
│   └── visualize_3d.py    # 3D PCA activation visualisation
│
├── scripts/
│   ├── run_mnist.sh                # Core MNIST experiments (Groups 1–2)
│   ├── run_fashionmnist.sh         # Core FashionMNIST experiments (Groups 1–2)
│   ├── run_noise_ablation.sh       # Noise + pretraining ablation at p=15%
│   ├── run_multilayer_eval.sh      # Multi-layer fusion experiments
│   ├── plot_multilayer_results.py  # Plots AC F1 vs poison rate per layer
│   └── plot_images.py              # Plots reconstruction figure for report
│
├── datasets/              # Generated and downloaded datasets 
├── checkpoints/           # Trained model checkpoints (auto-created)
├── logs/                  # Experiment logs and result CSVs (auto-created)
└── results/               # Per-experiment output figures and json (auto-created)
```

## Installation

```bash
pip install torch torchvision scikit-learn matplotlib numpy tqdm
```

Python 3.11 required.

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
./scripts/run_multilayer_eval.sh
```

**Key flags:**

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `MNIST` | `MNIST` or `FashionMNIST` |
| `--poison_rate` | `0.33` | Fraction of each class to poison |
| `--reconstruction_method` | `geiping` | `geiping` = cosine inversion, `dlg` = L2 inversion, `badnets` = no reconstruction |
| `--noise_std` | `0.0` | Gaussian noise added to intercepted gradients |
| `--layers` | `fc1` | Activation layers for AC, e.g. `conv1,fc1` |
| `--no_plots` | off | Suppress figure generation |

Datasets, poisoned caches, and model checkpoints are saved automatically
and reused on subsequent runs to avoid redundant computation.