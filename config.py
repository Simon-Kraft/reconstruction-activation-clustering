"""
config.py — Single source of truth for all hyperparameters.

To run a new experiment, change values here and re-run pipeline.py.
"""

import torch

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Dataset (MNIST)
# ---------------------------------------------------------------------------
MEAN = 0.1307
STD  = 0.3081

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1000

# ---------------------------------------------------------------------------
# Poisoning
# ---------------------------------------------------------------------------
# How many samples to pull from the training set, reconstruct via DLG,
# and replace with triggered+relabelled versions.
# These get mixed back into the full 60k training set.
N_POISON     = 50
TARGET_CLASS = 0            # class that receives the backdoor trigger

# Trigger stamp (pixel patch injected into poisoned images)
TRIGGER_SIZE = 3
TRIGGER_POS  = (24, 24)     # (row, col) of the top-left corner — bottom-right of 28×28
TRIGGER_VAL  = 2.8          # value written into the trigger pixels (in normalised space)

# ---------------------------------------------------------------------------
# Reconstruction model (DLG)
# ---------------------------------------------------------------------------
DLG_ITERATIONS       = 300   # L-BFGS steps per sample
DLG_LR               = 0.1
DLG_NOISE_STD        = 0.0   # Experiment B: try 0.01, 0.05, 0.1, 0.5


DLG_METHOD           = 'cosine'
DLG_TV_WEIGHT        = 1e-4  # total variation regularisation weight (cosine only)

# DLG_CLAMP            = (0.0, 1.0)
# After — correct normalised range matching the MNIST transform
DLG_CLAMP = ((0.0 - 0.1307) / 0.3081,   # ≈ -0.4242
              (1.0 - 0.1307) / 0.3081)   # ≈  2.8215

# How many epochs to pretrain the reconstruction model before running DLG.
# Experiment A: try 0 (untrained), 2, 5, 10
RECON_PRETRAIN_EPOCHS = 1

# ---------------------------------------------------------------------------
# Backdoor model training
# ---------------------------------------------------------------------------
TRAIN_EPOCHS = 10
TRAIN_LR     = 1e-3

# ---------------------------------------------------------------------------
# Activation Clustering
# ---------------------------------------------------------------------------
AC_N_COMPONENTS = 10        # PCA components before k-means
AC_N_CLUSTERS   = 2         # always 2 for the AC method
AC_LAYER        = 'fc1'     # which layer to use for detection (last hidden)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR        = 'data/'
CHECKPOINT_DIR  = 'checkpoints/'
RESULTS_DIR     = 'results/'

RECON_DATASET_PATH    = DATA_DIR + 'poisoned_recon_dataset.pt'
BACKDOOR_MODEL_PATH   = CHECKPOINT_DIR + 'backdoor_model.pt'
RESULTS_METRICS_PATH  = RESULTS_DIR + 'metrics.pt'

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL = 'LargeCNN'   # options: 'LargeCNN', 'MidCNN', 'SmallCNN'

N_SHOW       = 30   # total number of poisoned samples to display
COLS_PER_ROW = 10   # how many images per row