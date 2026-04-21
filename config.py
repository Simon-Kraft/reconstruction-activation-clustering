"""
config.py — Fixed hyperparameters and defaults.

Only contains values that are constant across experiments or rarely changed.
Everything that varies between runs (dataset, poison_rate, noise_std, etc.)
is controlled via argparse in pipeline.py.
"""

import torch
from data.builder                  import PoisonConfig
from clustering.analyzer import AnalysisConfig

# ---------------------------------------------------------------------------
# Device and reproducibility
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1000
DATASETS_DIR     = 'datasets/'
CHECKPOINT_DIR   = 'checkpoints/'

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TRAIN_EPOCHS = 10
TRAIN_LR     = 1e-3

# ---------------------------------------------------------------------------
# Activation Clustering
# ---------------------------------------------------------------------------
AC_N_COMPONENTS = 2
AC_METHOD       = 'ica'

# ---------------------------------------------------------------------------
# Runtime flags
# ---------------------------------------------------------------------------
SHOW_PLOTS = False

# ---------------------------------------------------------------------------
# Defaults — overridden by argparse in pipeline.py
# ---------------------------------------------------------------------------
DATASET_NAME = 'MNIST'
AC_LAYER     = 'fc1'

POISON_CFG = PoisonConfig(
    dataset_name       = DATASET_NAME,
    poison_rate        = 0.15,
    pretrain_epochs    = 0,
    dlg_iterations     = 75,
    dlg_lr             = 0.1,
    dlg_tv_weight      = 1e-4,
    noise_std          = 0.0,
    subsample_rate     = 0.25,
    data_dir           = DATASETS_DIR,
    seed               = SEED,
    use_reconstruction = True,
    replace_originals  = False,
)

ANALYSIS_CFG = AnalysisConfig(
    silhouette_threshold = 0.10,
    max_poison_rate      = POISON_CFG.poison_rate + 0.05,
    run_exre             = False,
    exre_threshold       = 1.0,
    exre_epochs          = 5,
    exre_lr              = 1e-3,
    seed                 = SEED,
)

# ---------------------------------------------------------------------------
# Paths — recomputed in pipeline.py after argparse overrides
# ---------------------------------------------------------------------------
BACKDOOR_MODEL_PATH = None
CACHE_DATASET_PATH  = None
RESULTS_DIR         = None