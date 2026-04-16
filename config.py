"""
config.py — Single source of truth for all hyperparameters.

Rotating poison setup matches Chen et al. (2018) exactly:
    class lm → class (lm+1) % n_classes for all lm simultaneously.
    All 10 MNIST classes are poisoned at the same poison_rate.

To run a different experiment, change values here and re-run pipeline.py.
Delete cached datasets in cache/ when changing any poison parameter.
"""

import torch
from data.builder import PoisonConfig
from activation_clustering.analyzer import AnalysisConfig

# ---------------------------------------------------------------------------
# Device and reproducibility
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED   = 42

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME     = 'MNIST'
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE  = 1000

# ---------------------------------------------------------------------------
# Poisoning — rotating scheme matching Chen et al. (2018)
# ---------------------------------------------------------------------------
POISON_CFG = PoisonConfig(
    dataset_name       = DATASET_NAME,
    poison_rate        = 0.33,
    pretrain_epochs    = 0,
    dlg_iterations     = 75,
    dlg_lr             = 0.1,
    dlg_tv_weight      = 1e-4,
    noise_std          = 0.0,
    subsample_rate     = 0.25,
    data_dir           = 'datasets/',
    seed               = SEED,
    use_reconstruction = True
)

# ---------------------------------------------------------------------------
# Backdoor model training
# ---------------------------------------------------------------------------
TRAIN_EPOCHS = 30
TRAIN_LR     = 1e-3

# ---------------------------------------------------------------------------
# Activation Clustering
# ---------------------------------------------------------------------------
AC_N_COMPONENTS = 10
AC_LAYERS        = ['fc1']
AC_METHOD       = 'ica'

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
# Runtime flags
# ---------------------------------------------------------------------------
SHOW_PLOTS      = False   # False = skip all visualisation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASETS_DIR   = 'datasets/'
CACHE_DIR      = 'datasets/'
CHECKPOINT_DIR = 'checkpoints/'

_EXP_ID = (
    f"{DATASET_NAME}"
    f"_rotating"
    f"_r{POISON_CFG.poison_rate}"
    f"_sub{POISON_CFG.subsample_rate}"
    f"_recon{int(POISON_CFG.use_reconstruction)}"
    f"_noise{POISON_CFG.noise_std}"
    f"_pre{POISON_CFG.pretrain_epochs}"
)

AC_LAYERS = sorted(AC_LAYERS)
CACHE_DATASET_PATH  = CACHE_DIR      + f'mixed_{_EXP_ID}.pt'
BACKDOOR_MODEL_PATH = CHECKPOINT_DIR + f'backdoor_model_{_EXP_ID}.pt'
_LAYER_ID = '+'.join(AC_LAYERS)
RESULTS_DIR = f'results/{_EXP_ID}_layers_{_LAYER_ID}/'